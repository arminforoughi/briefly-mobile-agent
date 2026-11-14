"""Phone Use primitives.

All agent logic talks to a `PhoneBackend` with a small, normalized set of
primitives:

    read_notifications · read_sms · read_screen · tap · type_text · swipe
    open_app · press_key · push_file · open_media · compose_sms

Three implementations:
  * MCPPhoneBackend  – routes primitives to a Phone Use MCP server (persistent
                       session via langchain-mcp-adapters); anything the server
                       does not expose falls back to ADB.
  * ADBPhoneBackend  – direct `adb` against a USB / Wi-Fi debugging device.
  * MockPhoneBackend – canned data for rehearsals and offline demos.
"""
from __future__ import annotations

import asyncio
import re
import shlex
import xml.etree.ElementTree as ET
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from .config import Config

PRIMITIVES = {
    "read_notifications", "read_sms", "read_screen", "tap", "type_text", "swipe",
    "open_app", "press_key", "push_file", "open_media", "compose_sms", "agent_run",
}

NOISY_PACKAGES = {"android", "com.android.systemui", "com.android.vending", "com.google.android.gms"}


@dataclass
class Notification:
    app: str
    title: str
    text: str

    def __str__(self) -> str:
        return f"[{self.app}] {self.title}: {self.text}"


@dataclass
class Sms:
    sender: str
    body: str
    unread: bool

    def __str__(self) -> str:
        return f"{'(unread) ' if self.unread else ''}{self.sender}: {self.body}"


@dataclass
class UiElement:
    text: str
    x: int
    y: int
    clickable: bool

    def __str__(self) -> str:
        return f"'{self.text}' @({self.x},{self.y}){' [btn]' if self.clickable else ''}"


class PhoneBackend:
    name = "base"
    on_screen = None  # callback(dict) for live UIs

    def _show(self, **view) -> None:
        if self.on_screen:
            self.on_screen(view)

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def read_notifications(self, limit: int = 15) -> list[Notification]: raise NotImplementedError
    async def read_sms(self, limit: int = 10) -> list[Sms]: raise NotImplementedError
    async def read_screen(self) -> str: raise NotImplementedError
    async def tap(self, x: int, y: int) -> str: raise NotImplementedError
    async def type_text(self, text: str) -> str: raise NotImplementedError
    async def swipe(self, x1: int, y1: int, x2: int, y2: int) -> str: raise NotImplementedError
    async def open_app(self, package: str) -> str: raise NotImplementedError
    async def press_key(self, key: str) -> str: raise NotImplementedError
    async def push_file(self, local: str, remote: str) -> str: raise NotImplementedError
    async def open_media(self, remote: str, mime: str = "audio/mpeg") -> str: raise NotImplementedError
    async def compose_sms(self, number: str, body: str) -> str: raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
#  ADB
# ═══════════════════════════════════════════════════════════════════════════
class ADBPhoneBackend(PhoneBackend):
    name = "adb"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def _adb(self, *args: str, timeout: float = 20) -> str:
        cmd = ["adb"] + (["-s", self.cfg.adb_serial] if self.cfg.adb_serial else []) + list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"adb {' '.join(args[:3])} failed: {err.decode(errors='replace').strip()}")
        return out.decode(errors="replace")

    async def _sh(self, command: str, timeout: float = 20) -> str:
        return await self._adb("shell", command, timeout=timeout)

    async def start(self) -> None:
        state = (await self._adb("get-state")).strip()
        if state != "device":
            raise RuntimeError(f"ADB device not ready (state={state!r})")

    # ── reads ───────────────────────────────────────────────────────────
    async def read_notifications(self, limit: int = 15) -> list[Notification]:
        raw = await self._sh("dumpsys notification --noredact")
        return parse_dumpsys_notifications(raw)[:limit]

    async def read_sms(self, limit: int = 10) -> list[Sms]:
        raw = await self._sh(
            "content query --uri content://sms/inbox --projection address:body:date:read "
            f"--sort {shlex.quote(f'date DESC LIMIT {int(limit)}')}"
        )
        return parse_sms_rows(raw)[:limit]

    async def read_screen(self) -> str:
        raw = await self._adb("exec-out", "uiautomator", "dump", "/dev/tty", timeout=30)
        return format_elements(parse_ui_dump(raw))

    # ── actions ─────────────────────────────────────────────────────────
    async def tap(self, x: int, y: int) -> str:
        await self._sh(f"input tap {int(x)} {int(y)}")
        return f"tapped ({x},{y})"

    async def type_text(self, text: str) -> str:
        await self._sh("input text " + shlex.quote(text.replace(" ", "%s")))
        return f"typed {len(text)} chars"

    async def swipe(self, x1: int, y1: int, x2: int, y2: int) -> str:
        await self._sh(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} 300")
        return "swiped"

    async def open_app(self, package: str) -> str:
        await self._sh(f"monkey -p {shlex.quote(package)} -c android.intent.category.LAUNCHER 1")
        await asyncio.sleep(1.5)
        return f"opened {package}"

    async def press_key(self, key: str) -> str:
        code = key.upper()
        code = code if code.startswith("KEYCODE_") else f"KEYCODE_{code}"
        await self._sh(f"input keyevent {shlex.quote(code)}")
        return f"pressed {code}"

    async def push_file(self, local: str, remote: str) -> str:
        await self._adb("push", local, remote, timeout=60)
        await self._sh(
            "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d " + shlex.quote(f"file://{remote}")
        )
        return f"pushed → {remote}"

    async def open_media(self, remote: str, mime: str = "audio/mpeg") -> str:
        await self._sh(
            "am start -a android.intent.action.VIEW -t " + shlex.quote(mime) + " -d " + shlex.quote(f"file://{remote}")
        )
        return f"playing {remote}"

    async def compose_sms(self, number: str, body: str) -> str:
        await self._sh(
            "am start -a android.intent.action.SENDTO -d " + shlex.quote(f"sms:{number}")
            + " --es sms_body " + shlex.quote(body) + " --ez exit_on_sent true"
        )
        await asyncio.sleep(1.5)
        return f"draft to {number} ({len(body)} chars)"


# ── parsers (pure functions, unit-testable) ─────────────────────────────────
_TITLE = re.compile(r"android\.title=\w+ \((.*)\)")
_TEXT = re.compile(r"android\.(?:bigText|text)=\w+ \((.*)\)")
_PKG = re.compile(r"pkg=(\S+)")


def parse_dumpsys_notifications(raw: str) -> list[Notification]:
    out, seen = [], set()
    for block in raw.split("NotificationRecord(")[1:]:
        pkg = _PKG.search(block)
        title = _TITLE.search(block)
        text = _TEXT.search(block)
        if not pkg or not (title or text):
            continue
        n = Notification(pkg.group(1), title.group(1) if title else "", text.group(1) if text else "")
        key = (n.app, n.title, n.text)
        if n.app in NOISY_PACKAGES or key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


_SMS_ROW = re.compile(r"address=(.*?), body=(.*), date=(\d+), read=(\d)\s*$")


def parse_sms_rows(raw: str) -> list[Sms]:
    rows, current = [], ""
    for line in raw.splitlines():
        if line.startswith("Row: "):
            if current:
                rows.append(current)
            current = line
        else:  # multi-line bodies
            current += "\n" + line
    if current:
        rows.append(current)
    out = []
    for r in rows:
        m = _SMS_ROW.search(r.replace("\n", " "))
        if m:
            out.append(Sms(m.group(1), m.group(2), m.group(4) == "0"))
    return out


_BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def parse_ui_dump(raw: str) -> list[UiElement]:
    start, end = raw.find("<?xml"), raw.rfind("</hierarchy>")
    if start < 0 or end < 0:
        return []
    root = ET.fromstring(raw[start : end + len("</hierarchy>")])
    els = []
    for node in root.iter("node"):
        label = node.get("text") or node.get("content-desc") or ""
        b = _BOUNDS.match(node.get("bounds", ""))
        if not label.strip() or not b:
            continue
        x1, y1, x2, y2 = map(int, b.groups())
        els.append(UiElement(label.strip(), (x1 + x2) // 2, (y1 + y2) // 2, node.get("clickable") == "true"))
    return els


def format_elements(els: list[UiElement], limit: int = 60) -> str:
    return "\n".join(f"{i}. {e}" for i, e in enumerate(els[:limit])) or "(empty screen)"


# ═══════════════════════════════════════════════════════════════════════════
#  Phone Use MCP
# ═══════════════════════════════════════════════════════════════════════════
class MCPPhoneBackend(PhoneBackend):
    """Routes primitives to MCP tools (persistent session), falls back to ADB."""

    name = "mcp"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.adb = ADBPhoneBackend(cfg)
        self.tools: dict[str, Any] = {}
        self._stack = AsyncExitStack()
        self.routed: dict[str, str] = {}

    async def start(self) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        if self.cfg.mcp_url:
            conn = {"transport": "streamable_http", "url": self.cfg.mcp_url}
        else:
            conn = {"transport": "stdio", "command": self.cfg.mcp_command[0], "args": self.cfg.mcp_command[1:]}
        client = MultiServerMCPClient({"phone": conn})
        session = await self._stack.enter_async_context(client.session("phone"))
        self.tools = {t.name: t for t in await load_mcp_tools(session)}
        for prim, spec in self.cfg.mcp_toolmap.items():
            self.routed[prim] = "mcp:" + spec["tool"] if spec.get("tool") in self.tools else "adb"
        # mobile-mcp style servers want a device selected first
        if "mobile_use_device" in self.tools and self.cfg.adb_serial:
            await self.tools["mobile_use_device"].ainvoke({"device": self.cfg.adb_serial, "deviceType": "android"})

    async def close(self) -> None:
        await self._stack.aclose()

    async def _route(self, prim: str, **kw) -> Any:
        spec = self.cfg.mcp_toolmap.get(prim)
        tool = self.tools.get(spec["tool"]) if spec else None
        if tool is None:
            return await getattr(self.adb, prim)(**kw)
        args = {
            k: (kw[v[1:-1]] if isinstance(v, str) and v.startswith("{") and v.endswith("}") else v)
            for k, v in spec.get("args", {}).items()
        }
        res = await tool.ainvoke(args)
        return _mcp_text(res)

    async def read_notifications(self, limit: int = 15):
        res = await self._route("read_notifications", limit=limit)
        return res if isinstance(res, list) else [Notification("mcp", "", line) for line in str(res).splitlines() if line.strip()]

    async def read_sms(self, limit: int = 10):
        res = await self._route("read_sms", limit=limit)
        return res if isinstance(res, list) else [Sms("?", line, True) for line in str(res).splitlines() if line.strip()]

    async def read_screen(self): return str(await self._route("read_screen"))
    async def tap(self, x, y): return await self._route("tap", x=x, y=y)
    async def type_text(self, text): return await self._route("type_text", text=text)
    async def swipe(self, x1, y1, x2, y2): return await self._route("swipe", x1=x1, y1=y1, x2=x2, y2=y2)
    async def open_app(self, package): return await self._route("open_app", package=package)
    async def press_key(self, key): return await self._route("press_key", key=key)
    async def push_file(self, local, remote): return await self.adb.push_file(local, remote)
    async def open_media(self, remote, mime="audio/mpeg"): return await self.adb.open_media(remote, mime)
    async def compose_sms(self, number, body): return await self.adb.compose_sms(number, body)


def _mcp_text(res: Any) -> Any:
    """Flatten MCP tool output (str | content blocks | (content, artifact)) to text."""
    if isinstance(res, tuple):
        res = res[0]
    if isinstance(res, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in res)
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  Mock (rehearsals / no device)
# ═══════════════════════════════════════════════════════════════════════════
class MockPhoneBackend(PhoneBackend):
    name = "mock"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.screen = "home"
        self.actions: list[str] = []

    async def read_notifications(self, limit=15):
        await asyncio.sleep(0.05)
        items = self._notifs()[:limit]
        self._show(view="shade", items=[{"app": n.app.split(".")[-1], "title": n.title, "text": n.text} for n in items])
        return items

    def _notifs(self):
        return [
            Notification("com.google.android.calendar", "Board sync moved", "Now 3:30 PM – 4:15 PM · Room Atlas (was 2:00 PM)"),
            Notification("com.Slack", "Priya Nair", "Can you skim the Q4 deck before the Nvidia partnership call? They'll ask about Blackwell supply"),
            Notification("com.google.android.gm", "Legal Ops", "Action required: EU AI Act GPAI obligations — sign-off needed by Friday"),
            Notification("com.google.android.apps.magazines", "Markets", "Fed holds rates steady, signals data-dependent path into 2027"),
            Notification("com.spotify.music", "Now playing", "Lo-fi beats"),
        ]

    async def read_sms(self, limit=10):
        await asyncio.sleep(0.05)
        items = self._sms()[:limit]
        self._show(view="messages", items=[{"from": s.sender, "text": s.body, "unread": s.unread} for s in items])
        return items

    def _sms(self):
        return [
            Sms("+1 415 555 0142 (Dana – boss)", "URGENT: client asked for EU AI Act compliance talking points, need them by 2pm", True),
            Sms("+1 650 555 0199 (Mom)", "Dinner Sunday still on? 🙂", True),
            Sms("+1 888 555 0100", "Your package was delivered", False),
        ]

    async def read_screen(self):
        await asyncio.sleep(0.05)
        if self.screen == self.cfg.email_app:
            self._show(view="gmail", items=[
                {"from": "Legal Ops", "subject": "EU AI Act GPAI obligations", "text": "Please review summary & sign off by Fri", "unread": True},
                {"from": "Jordan Lee", "subject": "Re: Offsite agenda", "text": "Moved to Dec 4, hotel confirmed", "unread": True},
                {"from": "Nvidia Partnerships", "subject": "Agenda for Thursday call", "text": "Supply roadmap, co-marketing", "unread": False},
            ])
            return format_elements([
                UiElement("Primary", 140, 300, True),
                UiElement("Legal Ops – EU AI Act GPAI obligations: please review summary & sign off by Fri", 540, 420, True),
                UiElement("Jordan Lee – Re: Offsite agenda — moved to Dec 4, hotel confirmed", 540, 560, True),
                UiElement("Nvidia Partnerships – Agenda for Thursday call: supply roadmap, co-marketing", 540, 700, True),
                UiElement("Compose", 900, 2100, True),
            ])
        if self.screen == "sms_compose":
            return format_elements([
                UiElement("To: " + (self.cfg.user_phone_number or "me"), 540, 200, False),
                UiElement("Text message (draft filled)", 480, 2050, True),
                UiElement("Send SMS", 990, 2050, True),
            ])
        return format_elements([UiElement("Home screen", 540, 1000, False)])

    async def _act(self, desc: str) -> str:
        await asyncio.sleep(0.03)
        self.actions.append(desc)
        return desc

    async def tap(self, x, y):
        if self.screen == "sms_compose" and x > 900:
            self.screen = "sms_sent"
            self._show(view="compose", to=self._draft[0], body=self._draft[1], sent=True)
            return await self._act(f"tapped ({x},{y}) – message sent")
        return await self._act(f"tapped ({x},{y})")

    async def type_text(self, text): return await self._act(f"typed {len(text)} chars")
    async def swipe(self, x1, y1, x2, y2): return await self._act("swiped")

    async def open_app(self, package):
        self.screen = package
        self._show(view="loading", app=package)
        return await self._act(f"opened {package}")

    async def press_key(self, key):
        if key.upper() in ("HOME", "KEYCODE_HOME"):
            self.screen = "home"
            self._show(view="home")
        return await self._act(f"pressed {key}")

    async def push_file(self, local, remote):
        self._show(view="transfer", file=remote.rsplit("/", 1)[-1])
        return await self._act(f"pushed → {remote}")

    async def open_media(self, remote, mime="audio/mpeg"):
        self._show(view="player", file=remote.rsplit("/", 1)[-1])
        return await self._act(f"playing {remote}")

    async def compose_sms(self, number, body):
        self.screen = "sms_compose"
        self._draft = (number, body)
        self._show(view="compose", to=number, body=body, sent=False)
        return await self._act(f"draft to {number} ({len(body)} chars)")


def make_backend(cfg: Config, llm=None) -> PhoneBackend:
    if cfg.phone_backend == "phoneuse":
        from .phoneuse import PhoneUseBackend

        return PhoneUseBackend(cfg, llm)
    return {"mcp": MCPPhoneBackend, "adb": ADBPhoneBackend, "mock": MockPhoneBackend}[cfg.phone_backend](cfg)


class TracedPhone:
    """Proxy that sends every primitive through the tracer (budget + logs)."""

    def __init__(self, inner: PhoneBackend, tracer):
        self._inner, self._tracer = inner, tracer

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if name not in PRIMITIVES:
            return attr

        async def traced(*args, **kwargs):
            return await self._tracer.step("PHONE", name, attr, *args, **kwargs)

        return traced
