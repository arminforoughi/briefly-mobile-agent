"""PhoneUse (phoneuse.com) backend — hosted MCP at https://mcp.phoneuse.com/mcp.

Differences from a generic Android MCP server that this adapter absorbs:
  * every tool needs `deviceId`; the device is leased (acquire → release)
  * coordinates are a normalized 0-1000 grid (not pixels)
  * cloud phones expose screenshots but no UI tree → screens are read with a
    vision LLM, which returns elements already in the 0-1000 grid
  * there is no SMS/notification API and no file push → those primitives are
    implemented as UI flows (pull down the shade, open Messages) or refused
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from .config import Config
from .phone import Notification, PhoneBackend, Sms

VISION_ELEMENTS = """You are the eyes of a phone-automation agent. List every readable UI element on this Android
screenshot, one per line, in exactly this format:
'<visible text or icon meaning>' @(<x>,<y>) [btn]
x and y are the element's CENTER on a 0-1000 grid (0,0 = top-left, 1000,1000 = bottom-right of the image).
Append [btn] only if it looks tappable. Top to bottom. No other text."""

VISION_NOTIFS = """This is the Android notification shade. Transcribe each notification on its own line as:
<app name> | <title or sender> | <message text>
Skip system/ongoing items (USB, battery, media controls). If there are none, reply NONE."""

VISION_SMS = """This is a messaging app's conversation list. Transcribe each conversation on its own line as:
<sender> | <latest message preview> | <unread: yes/no (bold = unread)>
If there are none, reply NONE."""


_SENDER = re.compile(r"^(\+?\(?\d[\d\s()\-]{6,}|[A-Z][\w .'-]{1,30})$")


def group_conversations(lines: list[str]) -> list[Sms]:
    """OCR of a conversation list gives 'sender' then 'preview' lines; pair them up."""
    out, i = [], 0
    while i < len(lines):
        if _SENDER.match(lines[i]) and i + 1 < len(lines) and not _SENDER.match(lines[i + 1]):
            out.append(Sms(lines[i], lines[i + 1], False))
            i += 2
        else:
            out.append(Sms("?", lines[i], False))
            i += 1
    return out


class PhoneUseBackend(PhoneBackend):
    name = "phoneuse"

    def __init__(self, cfg: Config, llm: Any = None):
        self.cfg, self.llm = cfg, llm
        self.device = cfg.device_id
        self.lease: str | None = None
        self._stack = AsyncExitStack()
        self._session = None
        self.shots_dir: Path | None = None
        self._shot_n = 0
        self.routed: dict[str, str] = {}

    # ── session / lease ────────────────────────────────────────────────
    async def start(self) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        url = self.cfg.mcp_url or "https://mcp.phoneuse.com/mcp"
        client = MultiServerMCPClient({"phoneuse": {
            "transport": "streamable_http", "url": url,
            "headers": {"Authorization": f"Bearer {self.cfg.mcp_api_key}"},
        }})
        self._session = await self._stack.enter_async_context(client.session("phoneuse"))
        devices = (await self._call("list_devices", with_device=False))["devices"]
        if not self.device:
            online = [d for d in devices if d.get("status") == "online"]
            if not online:
                raise RuntimeError("no online PhoneUse device")
            self.device = online[0]["deviceId"]
        dev = next((d for d in devices if d["deviceId"] == self.device), None)
        if not dev:
            raise RuntimeError(f"device {self.device} not visible to this API key")
        caps = dev["capabilities"]
        info = dev.get("info", {})
        lease = await self._call("acquire_device")
        self.lease = lease.get("leaseId")
        await self._call("wake")
        ui_tree = caps["observations"].get("uiTree")
        self.routed = {
            "device": f"{info.get('model', '?')} / Android {info.get('androidVersion', '?')} ({dev.get('kind')})",
            "read_screen": "mcp:ui_snapshot" if ui_tree else "mcp:screenshot+vision",
            "read_notifications": "ui:shade+vision", "read_sms": "ui:messages+vision",
            "tap/swipe/type/key/open_app": "mcp", "push_file": "unsupported",
        }
        self._ui_tree = bool(ui_tree)

    async def close(self) -> None:
        try:
            if self.lease and self._session:
                await self._call("release_device", leaseId=self.lease)
        finally:
            await self._stack.aclose()

    async def _call(self, tool: str, with_device: bool = True, **args) -> dict:
        if with_device:
            args = {"deviceId": self.device, **args}
        for attempt in range(4):
            res = await self._session.call_tool(tool, args)
            if not res.isError:
                break
            text = " ".join(getattr(b, "text", "") for b in res.content)
            if '"retryable":true' in text and attempt < 3:  # e.g. device_busy
                await asyncio.sleep(2 + attempt * 2)
                continue
            raise RuntimeError(f"{tool}: {text[:300]}")
        out = dict(res.structuredContent or {})
        imgs = [b for b in res.content if b.type == "image"]
        if imgs:
            out["_image"] = imgs[0]
        if not out:
            for b in res.content:
                if getattr(b, "text", None):
                    try:
                        out = json.loads(b.text)
                    except json.JSONDecodeError:
                        out = {"text": b.text}
        return out

    # ── observation ────────────────────────────────────────────────────
    async def _screenshot(self) -> tuple[str, str]:
        res = await self._call("screenshot")
        img = res["_image"]
        self._shot_n += 1
        if self.shots_dir:
            self.shots_dir.mkdir(parents=True, exist_ok=True)
            ext = "jpg" if "jpeg" in img.mimeType else "png"
            path = self.shots_dir / f"{self._shot_n:02d}.{ext}"
            path.write_bytes(base64.b64decode(img.data))
            try:
                url = "/" + str(path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                url = None
            if url:
                self._show(view="image", url=url)
        return img.data, img.mimeType

    async def _ocr(self) -> list[tuple[str, int, int]]:
        from .local_fallbacks import ocr_elements

        data, _ = await self._screenshot()
        return await asyncio.to_thread(ocr_elements, base64.b64decode(data))

    async def _see(self, prompt: str) -> str:
        if not self.llm:  # keyless: on-device OCR, same output format as the vision prompt
            return "\n".join(f"'{t}' @({x},{y})" for t, x, y in await self._ocr()) or "(empty screen)"
        data, mime = await self._screenshot()
        msg = await self.llm.ainvoke([{"role": "user", "content": [
            {"type": "image", "source_type": "base64", "data": data, "mime_type": mime},
            {"type": "text", "text": prompt},
        ]}])
        text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content if isinstance(b, dict))
        return text.strip()

    async def read_screen(self) -> str:
        if self._ui_tree:
            return json.dumps(await self._call("ui_snapshot"))[:6000]
        return await self._see(VISION_ELEMENTS)

    async def read_notifications(self, limit: int = 15) -> list[Notification]:
        await self._call("swipe", fromX=500, fromY=5, toX=500, toY=750, durationMs=300)
        await asyncio.sleep(0.8)
        try:
            if not self.llm:
                from .local_fallbacks import content_lines

                return [Notification("shade", "", t) for t in content_lines(await self._ocr())][:limit]
            text = await self._see(VISION_NOTIFS)
        finally:
            await self._call("press_key", key="back")
        out = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                out.append(Notification(parts[0], parts[1], parts[2]))
        return out[:limit]

    async def _open_sms_app(self) -> str:
        last = None
        for pkg in dict.fromkeys([self.cfg.sms_app, "com.google.android.apps.messaging", "com.android.messaging"]):
            try:
                await self._call("open_app", appId=pkg)
                return pkg
            except RuntimeError as e:
                last = e
        raise RuntimeError(f"no messaging app installed ({last})")

    async def read_sms(self, limit: int = 10) -> list[Sms]:
        await self._open_sms_app()
        await asyncio.sleep(1.5)
        try:
            if not self.llm:
                from .local_fallbacks import content_lines

                lines = [t for t in content_lines(await self._ocr())
                         if t.lower() not in ("draft", "+") and not t.lower().startswith(("once you start", "listed here"))]
                return group_conversations(lines)[:limit]
            text = await self._see(VISION_SMS)
        finally:
            await self._call("press_key", key="home")
        out = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                out.append(Sms(parts[0], parts[1], len(parts) > 2 and parts[2].lower().startswith("y")))
        return out[:limit]

    # ── actions (0-1000 grid) ──────────────────────────────────────────
    async def tap(self, x: int, y: int) -> str:
        await self._call("tap", x=x, y=y)
        return f"tapped ({x},{y})"

    async def type_text(self, text: str) -> str:
        await self._call("type_text", text=text)
        return f"typed {len(text)} chars"

    async def swipe(self, x1: int, y1: int, x2: int, y2: int) -> str:
        await self._call("swipe", fromX=x1, fromY=y1, toX=x2, toY=y2)
        return "swiped"

    async def open_app(self, package: str) -> str:
        await self._call("open_app", appId=package)
        await asyncio.sleep(1.5)
        return f"opened {package}"

    async def press_key(self, key: str) -> str:
        k = key.lower().removeprefix("keycode_")
        await self._call("press_key", key=k)
        return f"pressed {k}"

    async def push_file(self, local: str, remote: str) -> str:
        raise NotImplementedError("PhoneUse has no file-push API — use DELIVER=sms or call")

    async def open_media(self, remote: str, mime: str = "audio/mpeg") -> str:
        raise NotImplementedError("PhoneUse has no file-push API")

    async def agent_run(self, goal: str, max_steps: int = 25, timeout_s: int = 240) -> str:
        """Hand a natural-language goal to PhoneUse's hosted agent and wait (bounded) for it to finish."""
        run = await self._call("start_run", goal=goal, executor="phoneuse", maxSteps=max_steps,
                               deadlineMs=timeout_s * 1000)
        state, rid = run["state"], run["runId"]
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout_s
        info = run
        while state in ("queued", "running") and loop.time() < end:
            info = await self._call("get_run", with_device=False, runId=rid, waitMs=6000, sinceState=state)
            state = info["state"]
            if self.on_screen and state in ("queued", "running"):
                try:  # mirror the hosted agent's progress for live UIs
                    await self._screenshot()
                except RuntimeError:
                    pass
        if state in ("queued", "running"):
            await self._call("cancel_run", with_device=False, runId=rid)
            state = "canceled (timeout)"
        self.last_run = info
        return f"run {rid[:8]} {state} after {info.get('stepSeq', '?')} steps"

    async def compose_sms(self, number: str, body: str) -> str:
        # No intent API: open Messages; the UI agent then starts the chat, types and sends.
        pkg = await self._open_sms_app()
        await asyncio.sleep(1.5)
        return f"opened {pkg}"
