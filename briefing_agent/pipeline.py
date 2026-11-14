"""The briefing workflow: MONITOR → EXTRACT → RESEARCH → SCRIPT → TTS → DELIVER.

Stability strategy (fits in 50 steps):
  * The outer workflow is a deterministic pipeline, not one open-ended agent —
    the LLM is used where judgement is needed (extraction, research synthesis,
    script writing, the UI-driving sub-agent), never to decide control flow.
  * Every phone/LLM/web/TTS call costs one step from a shared hard budget.
  * The only free-form agent (UI navigation for SMS send) receives a
    LangGraph recursion_limit derived from the *remaining* budget, plus a
    deterministic fallback if it fails.
  * Every external dependency degrades gracefully (no key → heuristic/skip).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .config import Config
from .phone import TracedPhone
from .tracer import BudgetExceeded, Tracer


# ── structured outputs ──────────────────────────────────────────────────────
class ActionItem(BaseModel):
    source: Literal["notification", "sms", "email"]
    sender: str = Field(description="person or app it came from")
    summary: str = Field(description="one sentence, what it means for the user")
    kind: Literal["action", "schedule_change", "urgent", "info"]
    priority: Literal["high", "medium", "low"]
    due: str | None = Field(default=None, description="deadline/time if any, verbatim")


class Digest(BaseModel):
    items: list[ActionItem] = Field(description="actionable / schedule / high-priority items only; drop noise")
    research_topics: list[str] = Field(
        description="up to 3 web-searchable topics (meeting context, companies, regulations, news) "
        "where background would help the user. Phrase as search queries."
    )

    def summary_line(self) -> str:
        hi = sum(i.priority == "high" for i in self.items)
        return f"{len(self.items)} items ({hi} high) · {len(self.research_topics)} research topics"


class ResearchNote(BaseModel):
    topic: str
    bullets: list[str] = Field(description="3-4 crisp factual bullets, latest first")
    sources: list[str] = Field(default_factory=list)

    def summary_line(self) -> str:
        return f"{len(self.bullets)} bullets · {len(self.sources)} sources"


@dataclass
class BriefResult:
    digest: Digest
    research: list[ResearchNote] = field(default_factory=list)
    script: str = ""
    audio_path: str | None = None
    delivered: list[str] = field(default_factory=list)


# ── LLM helpers ─────────────────────────────────────────────────────────────
_LLM_PROBE: dict[str, str | None] = {}  # model → None (usable) or the reason it isn't


def llm_status(cfg: Config) -> str | None:
    """Why the configured LLM can't be used right now (no key, no credits, bad model…), or None if it can."""
    return _LLM_PROBE.get(cfg.model)


def make_llm(cfg: Config):
    if not cfg.llm_enabled:
        return None
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(cfg.model, max_retries=2, timeout=60)
    if cfg.model not in _LLM_PROBE:  # one tiny call per process, so a dead key can't break a live demo
        try:
            llm.bind(max_tokens=1).invoke("ping")
            _LLM_PROBE[cfg.model] = None
        except Exception as e:
            msg = str(getattr(e, "message", "") or e)
            m = re.search(r"'message': '([^']+)'", msg)
            _LLM_PROBE[cfg.model] = (m.group(1) if m else msg)[:140]
    return None if _LLM_PROBE[cfg.model] else llm


EXTRACT_PROMPT = """You are the triage brain of a mobile briefing agent. The user is away from their screen.
From the raw phone data below, extract ONLY what matters: actionable requests, schedule changes,
deadlines, and high-priority messages. Ignore media players, promos, delivery spam, OTP codes.
Pick up to {k} research topics where web background would make the user sound informed
(e.g. a company they're meeting, a regulation mentioned, a news event).

## Notifications
{notifs}

## SMS (newest first)
{sms}

## Email inbox (screen read)
{email}
"""

SCRIPT_PROMPT = """Write a spoken voice briefing (60–90 seconds, ~170 words) for the user, who is away from their screen.
Style: warm, confident, like a sharp chief-of-staff. Lead with the most urgent item. Group schedule changes.
Weave in 1–2 research facts where they help the user act. End with a one-line "top 3 to-dos" recap.
No markdown, no emojis, no URLs — this will be read aloud by text-to-speech. Say times naturally.
Current local time: {now}

## Triage
{items}

## Research
{research}
"""

SMS_TEXT_PROMPT = "Compress this briefing into one SMS under 300 characters, plain text, most urgent first:\n\n{script}"


# ── canned research for offline mock demos (clearly labelled as mock) ───────
MOCK_RESEARCH = {
    "eu": ResearchNote(topic="EU AI Act", sources=["mock://eu-ai-act"], bullets=[
        "General-purpose AI model obligations have applied since August 2025.",
        "Providers must keep technical documentation and publish a training-data summary.",
        "A voluntary GPAI Code of Practice is the main route to show compliance.",
    ]),
    "nvidia": ResearchNote(topic="Nvidia", sources=["mock://nvidia"], bullets=[
        "Blackwell-generation supply remains the gating factor for most partners.",
        "Expect questions on allocation timelines and co-marketing commitments.",
    ]),
    "fed": ResearchNote(topic="Fed", sources=["mock://fed"], bullets=[
        "Rates held steady; guidance stays data-dependent.",
    ]),
}


# ── heuristic fallbacks (no LLM key / LLM failure) ──────────────────────────
_URGENT = re.compile(r"\b(urgent|asap|action required|by \d|deadline|sign[- ]off|need)\b", re.I)
_SCHED = re.compile(r"\b(moved|resched|now \d|postpon|cancel|agenda|call|meeting|offsite)\b", re.I)
_NOISE = re.compile(r"(now playing|delivered|otp|verification code|promo|your briefing)", re.I)


def heuristic_digest(notifs, sms, email_screen: str) -> Digest:
    items: list[ActionItem] = []
    raw = [("notification", APP_NAMES.get(n.app.split(".")[-1].lower(), n.app.split(".")[-1]), f"{n.title}: {n.text}")
           for n in notifs]
    raw += [("sms", s.sender, s.body) for s in sms]
    raw += [("email", "Gmail", line.split("'")[1]) for line in email_screen.splitlines() if " – " in line]
    for src, who, text in raw:
        if _NOISE.search(text):
            continue
        urgent, sched = bool(_URGENT.search(text)), bool(_SCHED.search(text))
        if not (urgent or sched):
            continue
        items.append(ActionItem(source=src, sender=who, summary=text[:160],
                                kind="urgent" if urgent else "schedule_change",
                                priority="high" if urgent else "medium"))
    items = _dedupe(items)
    topics = []
    blob = " ".join(t for _, _, t in raw)
    for pat, query in ((r"EU A[Il1] Act", "EU AI Act GPAI obligations"), (r"Nvidia", "Nvidia Blackwell supply outlook"),
                       (r"\bFed\b", "Fed rate decision outlook")):
        if re.search(pat, blob, re.I):
            topics.append(query)
    return Digest(items=items, research_topics=topics)


_STOP = set("the a an to of and or for by on in at with your you is are be it this that can before need".split())
_PREFIX = re.compile(r"^(urgent( from \w+)?|action required|re|[A-Z][a-z]+ (?:[A-Z][a-z]+|Ops))\s*[:–—-]\s+", re.I)
APP_NAMES = {"gm": "Gmail", "calendar": "Calendar", "magazines": "News", "slack": "Slack"}
_NUM = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


def _dedupe(items: list[ActionItem]) -> list[ActionItem]:
    """Merge near-duplicates (same story arriving as notification + email + SMS); keep the highest priority."""
    rank = {"high": 0, "medium": 1, "low": 2}
    kept: list[ActionItem] = []
    for it in sorted(items, key=lambda i: rank[i.priority]):
        w = _words(it.summary)
        if any(len(w & _words(k.summary)) >= max(3, len(w) // 3) for k in kept):
            continue
        kept.append(it)
    return kept


def _spoken(summary: str) -> str:
    s = summary.strip().rstrip(".")
    for _ in range(2):  # drop "Legal Ops: Action required: …" style prefixes
        s = _PREFIX.sub("", s, count=1)
    s = s.replace(" — ", ", ").replace(" · ", ", ").replace(" – ", " to ")
    return s[:1].upper() + s[1:] + "."


def template_script(d: Digest, research: list[ResearchNote]) -> str:
    hi = [i for i in d.items if i.priority == "high"]
    rest = [i for i in d.items if i.priority != "high"]
    if not d.items:
        return ("Hey, quick check-in from your phone. All quiet: no urgent messages, no schedule changes, "
                "and nothing that needs you right now. Enjoy the break.")
    n = len(hi)
    parts = ["Hey, here's your briefing."]
    if hi:
        parts.append(f"{'One thing needs' if n == 1 else _NUM.get(n, str(n)) + ' things need'} you first.")
        for lead, it in zip(("First,", "Second,", "And third,"), hi[:3]):
            who = it.sender.split("(")[-1].rstrip(")").split(" –")[0]
            parts.append(f"{lead} from {who}: {_spoken(it.summary)}")
    if rest:
        parts.append("On your schedule: " + " ".join(_spoken(i.summary) for i in rest[:3]))
    for r in research[:2]:
        if r.bullets:
            parts.append(f"Quick context on {r.topic}: {r.bullets[0]}")
    parts.append("That's it. I'll ping you if anything else lands.")
    return " ".join(parts)


# ── the pipeline ────────────────────────────────────────────────────────────
class BriefingPipeline:
    def __init__(self, cfg: Config, phone: TracedPhone, tracer: Tracer, llm=None):
        self.cfg, self.phone, self.t = cfg, phone, tracer
        self.llm = llm if llm is not None else make_llm(cfg)
        self.run_dir = tracer.run_dir

    # 1 ─ MONITOR & EXTRACT ───────────────────────────────────────────────
    async def monitor(self) -> Digest:
        self.t.phase("PHASE 1 · MONITOR & EXTRACT")
        notifs = await self.phone.read_notifications(limit=15)
        sms = []
        try:
            sms = await self.phone.read_sms(limit=10)
        except BudgetExceeded:
            raise
        except Exception as e:  # SMS is best-effort too
            self.t.warn(f"SMS read failed, continuing: {str(e)[:120]}")
        email_screen = ""
        try:
            await self.phone.open_app(self.cfg.email_app)
            email_screen = await self.phone.read_screen()
            await self.phone.press_key("HOME")
        except BudgetExceeded:
            raise
        except Exception as e:  # email is best-effort
            self.t.warn(f"inbox read failed, continuing with notifications+SMS: {e}")

        for n in notifs[:6]:
            self.t.note(f"🔔 {n}")
        for s in sms[:4]:
            self.t.note(f"💬 {s}")

        if self.llm:
            prompt = EXTRACT_PROMPT.format(
                k=self.cfg.max_research_topics,
                notifs="\n".join(map(str, notifs)) or "(none)",
                sms="\n".join(map(str, sms)) or "(none)",
                email=email_screen or "(unavailable)",
            )
            try:
                digest = await self.t.step("LLM", "extract_digest",
                                           self.llm.with_structured_output(Digest).ainvoke, prompt)
            except BudgetExceeded:
                raise
            except Exception as e:
                self.t.warn(f"LLM extraction failed → heuristic triage ({e})")
                digest = await self.t.step("LLM", "heuristic_digest", heuristic_digest, notifs, sms, email_screen)
        else:
            digest = await self.t.step("LLM", "heuristic_digest", heuristic_digest, notifs, sms, email_screen)

        digest.research_topics = digest.research_topics[: self.cfg.max_research_topics]
        for it in digest.items:
            icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}[it.priority]
            self.t.note(f"{icon} [{it.kind}] {it.sender}: {it.summary}" + (f" (due {it.due})" if it.due else ""))
        return digest

    # 2 ─ DEEP RESEARCH ───────────────────────────────────────────────────
    async def research(self, digest: Digest) -> list[ResearchNote]:
        self.t.phase("PHASE 2 · DEEP RESEARCH (Tavily)")
        if not digest.research_topics:
            self.t.note("no research topics — skipping")
            return []
        if not self.cfg.tavily_api_key and self.cfg.phone_backend == "mock":
            self.t.note("mock mode → canned research notes (offline-safe)")
            notes = []
            for topic in digest.research_topics:
                def lookup(q: dict) -> ResearchNote:
                    key = q["query"].split()[0].lower()
                    return MOCK_RESEARCH.get(key, ResearchNote(topic=q["query"], bullets=["(no canned note)"])).model_copy()

                note = await self.t.step("WEB", "mock_search", lookup, {"query": topic})
                note.topic = topic
                notes.append(note)
            for n in notes:
                self.t.note(f"🔎 {n.topic}")
            return notes
        if self.cfg.tavily_api_key:
            from langchain_tavily import TavilySearch

            search = TavilySearch(max_results=4, search_depth="advanced", include_answer=True,
                                  tavily_api_key=self.cfg.tavily_api_key)
            engine, search_fn = "tavily_search", search.ainvoke
        else:
            from .local_fallbacks import web_search

            self.t.note("no TAVILY_API_KEY → keyless DuckDuckGo search")
            engine = "ddg_search"

            async def search_fn(q):
                return await asyncio.to_thread(web_search, q["query"])

        async def one(topic: str) -> ResearchNote | None:
            try:
                raw = await self.t.step("WEB", engine, search_fn, {"query": topic})
            except BudgetExceeded:
                raise
            except Exception as e:
                self.t.warn(f"search failed for {topic!r}: {e}")
                return None
            if isinstance(raw, str):
                raw = json.loads(raw) if raw.strip().startswith("{") else {"answer": raw, "results": []}
            results = raw.get("results", [])
            sources = [r.get("url", "") for r in results][:4]
            if self.llm:
                ctx = (raw.get("answer") or "") + "\n\n" + "\n\n".join(
                    f"{r.get('title')}\n{r.get('content', '')[:800]}" for r in results)
                try:
                    note = await self.t.step(
                        "LLM", "summarize_research",
                        self.llm.with_structured_output(ResearchNote).ainvoke,
                        f"Topic: {topic}\nSummarize into 3-4 crisp, current, factual bullets for a busy "
                        f"executive. Only use facts from the context.\n\n{ctx}")
                    note.topic, note.sources = topic, sources
                    return note
                except BudgetExceeded:
                    raise
                except Exception as e:
                    self.t.warn(f"research summarization failed: {e}")
            bullets = [raw["answer"]] if raw.get("answer") else []
            bullets += [(r.get("title", "") + " — " + r.get("content", "")[:140]).strip(" —") for r in results[:3]]
            return ResearchNote(topic=topic, bullets=bullets, sources=sources)

        # research topics in parallel — each still costs its own steps
        notes = [n for n in await asyncio.gather(*(one(t) for t in digest.research_topics)) if n]
        for n in notes:
            self.t.note(f"🔎 {n.topic}")
            for b in n.bullets:
                self.t.note(f"   • {b}")
        return notes

    # 3 ─ SCRIPT + TTS ────────────────────────────────────────────────────
    async def write_script(self, digest: Digest, research: list[ResearchNote]) -> str:
        self.t.phase("PHASE 3 · VOICE BRIEFING (script + ElevenLabs)")
        if self.llm:
            prompt = SCRIPT_PROMPT.format(
                now=datetime.now().strftime("%A %I:%M %p"),
                items="\n".join(f"- [{i.priority}/{i.kind}] {i.sender}: {i.summary} (due {i.due})" for i in digest.items),
                research="\n".join(f"- {r.topic}: " + " | ".join(r.bullets) for r in research) or "(none)",
            )
            try:
                msg = await self.t.step("LLM", "write_script", self.llm.ainvoke, prompt)
                script = msg.content if isinstance(msg.content, str) else "".join(
                    b.get("text", "") for b in msg.content if isinstance(b, dict))
            except BudgetExceeded:
                raise
            except Exception as e:
                self.t.warn(f"script LLM failed → template ({e})")
                script = template_script(digest, research)
        else:
            script = await self.t.step("LLM", "template_script", template_script, digest, research)
        script = script.strip()
        (self.run_dir / "brief.txt").write_text(script, encoding="utf-8")
        self.t.note(f"📝 {len(script.split())} words → {self.run_dir / 'brief.txt'}")
        return script

    async def synthesize(self, script: str) -> str | None:
        if not self.cfg.elevenlabs_api_key:
            from .local_fallbacks import kokoro_available, kokoro_tts, say_tts

            try:
                if kokoro_available():
                    self.t.note(f"no ELEVENLABS_API_KEY → local Kokoro-82M voice ({self.cfg.kokoro_voice})")
                    out = await self.t.step("TTS", "kokoro_tts", asyncio.to_thread, kokoro_tts, script,
                                            self.run_dir / "brief", self.cfg.kokoro_voice)
                else:
                    self.t.note("no ELEVENLABS_API_KEY → on-device macOS voice")
                    out = await self.t.step("TTS", "macos_say", asyncio.to_thread, say_tts, script, self.run_dir / "brief")
                return str(out)
            except BudgetExceeded:
                raise
            except Exception as e:
                self.t.warn(f"local TTS failed: {e}")
                return None
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self.cfg.elevenlabs_api_key)
        out = self.run_dir / "brief.mp3"

        def _tts() -> str:
            audio = client.text_to_speech.convert(
                text=script, voice_id=self.cfg.elevenlabs_voice_id,
                model_id=self.cfg.elevenlabs_model, output_format="mp3_44100_128")
            with out.open("wb") as f:
                for chunk in audio:
                    f.write(chunk)
            return f"{out} ({out.stat().st_size // 1024} KB)"

        try:
            await self.t.step("TTS", "elevenlabs_tts", asyncio.to_thread, _tts)
            return str(out)
        except BudgetExceeded:
            raise
        except Exception as e:
            self.t.warn(f"TTS failed: {e}")
            return None

    # 4 ─ DELIVER ─────────────────────────────────────────────────────────
    async def deliver(self, script: str, audio: str | None) -> list[str]:
        self.t.phase("PHASE 4 · PHONE DELIVERY")
        done: list[str] = []
        if self.cfg.dry_run:
            self.t.warn("DRY_RUN — not touching the phone / placing calls")
            return done
        for mode in self.cfg.deliver:
            try:
                if mode == "audio" and audio:
                    remote = f"{self.cfg.phone_audio_dir}/briefing_{self.run_dir.name}.mp3"
                    await self.phone.push_file(audio, remote)
                    await self.phone.open_media(remote)
                    done.append("audio")
                elif mode == "sms":
                    if await self._deliver_sms(script):
                        done.append("sms")
                elif mode == "phoneuse":
                    if await self._deliver_phoneuse(script):
                        done.append("phoneuse-agent")
                elif mode == "call":
                    if await self._deliver_call(script):
                        done.append("call")
                elif mode == "audio":
                    self.t.warn("audio delivery skipped (no mp3)")
            except BudgetExceeded:
                raise
            except Exception as e:
                self.t.warn(f"delivery '{mode}' failed: {e}")
        return done

    async def _deliver_sms(self, script: str) -> bool:
        number = self.cfg.user_phone_number
        if not number:
            self.t.warn("USER_PHONE_NUMBER not set — skipping SMS")
            return False
        text = script[:300]
        if self.llm:
            try:
                msg = await self.t.step("LLM", "compress_sms", self.llm.ainvoke, SMS_TEXT_PROMPT.format(script=script))
                text = str(msg.content).strip()[:320]
            except BudgetExceeded:
                raise
            except Exception:
                pass
        await self.phone.compose_sms(number, text)
        return await UiAgent(self).run(
            goal=f"Send one SMS to {number} with exactly this text:\n{text}\n\n"
                 "If a draft to that number is already on screen, just tap Send. Otherwise start a new chat "
                 "with that number, tap the message field, type the text, and tap Send exactly once. "
                 "Read the screen again to confirm it was sent.",
            fallback_label="send",
        )

    async def _deliver_phoneuse(self, script: str) -> bool:
        """Delegate the on-phone delivery to PhoneUse's own hosted agent (no LLM key needed)."""
        if not hasattr(self.phone._inner, "agent_run"):
            self.t.warn("phoneuse delivery needs PHONE_BACKEND=phoneuse")
            return False
        brief = re.sub(r"\s+", " ", script)[:600]
        number = self.cfg.user_phone_number
        number = number or "4155550100"  # fictional 555 number for drafts
        if self.cfg.deliver_send:
            goal = (f"Open the Messaging app, start a new conversation with {number}, type exactly this text "
                    f"into the message field and send it once:\n{brief}")
        else:
            goal = (f"Open the Messaging app, tap the + button, type {number} in the To field and press enter. "
                    f"Tap the 'Send message' field and type exactly this text:\n{brief}\n"
                    "Do NOT tap send. Finally tap the back arrow at the top-left so it is saved as a draft.")
        self.t.note(f"🤖 handing delivery to PhoneUse agent → {'SMS' if self.cfg.deliver_send else 'draft'} to {number}")
        res = await self.phone.agent_run(goal)
        return "succeeded" in res

    async def _deliver_call(self, script: str) -> bool:
        c = self.cfg
        if not (c.twilio_sid and c.twilio_token and c.twilio_from and c.user_phone_number):
            self.t.warn("Twilio not configured — skipping voice call")
            return False
        from xml.sax.saxutils import escape

        from twilio.rest import Client

        if c.public_audio_base_url:
            twiml = f"<Response><Play>{c.public_audio_base_url.rstrip('/')}/{self.run_dir.name}/brief.mp3</Play></Response>"
        else:
            twiml = f"<Response><Say voice=\"Polly.Joanna\">{escape(script)}</Say></Response>"
        client = Client(c.twilio_sid, c.twilio_token)
        await self.t.step("DELIVER", "twilio_voice_call", asyncio.to_thread,
                          lambda: client.calls.create(to=c.user_phone_number, from_=c.twilio_from, twiml=twiml).sid)
        return True

    # ── full cycle ──────────────────────────────────────────────────────
    async def run(self, state: "SeenState | None" = None, force: bool = False) -> BriefResult | None:
        digest = await self.monitor()
        if state is not None and not force:
            fresh = state.filter_new(digest)
            if not fresh.items:
                self.t.note("nothing new since last briefing — standing by")
                return None
            digest = fresh
        self.t.emit("digest", items=[i.model_dump() for i in digest.items], topics=digest.research_topics)
        notes = await self.research(digest)
        self.t.emit("research", notes=[n.model_dump() for n in notes])
        script = await self.write_script(digest, notes)
        self.t.emit("script", text=script)
        audio = await self.synthesize(script)
        if audio:
            self.t.emit("audio", url="/" + audio)
        delivered = await self.deliver(script, audio)
        self.t.emit("delivered", modes=delivered)
        if state is not None:
            state.mark(digest)
        return BriefResult(digest, notes, script, audio, delivered)


# ── bounded UI sub-agent (Phone Use primitives as LangChain tools) ──────────
class UiAgent:
    """ReAct agent over phone primitives with a recursion limit tied to the remaining step budget."""

    MAX_UI_STEPS = 8

    def __init__(self, pipe: BriefingPipeline):
        self.p = pipe

    def _tools(self):
        from langchain_core.tools import tool

        phone = self.p.phone

        @tool
        async def read_screen() -> str:
            """Read the visible UI elements with their tap coordinates."""
            return await phone.read_screen()

        @tool
        async def tap(x: int, y: int) -> str:
            """Tap the screen at pixel coordinates (x, y)."""
            return await phone.tap(x, y)

        @tool
        async def type_text(text: str) -> str:
            """Type text into the focused field."""
            return await phone.type_text(text)

        @tool
        async def swipe(x1: int, y1: int, x2: int, y2: int) -> str:
            """Swipe from (x1,y1) to (x2,y2) to scroll."""
            return await phone.swipe(x1, y1, x2, y2)

        @tool
        async def press_key(key: str) -> str:
            """Press a hardware key: BACK, HOME, ENTER."""
            return await phone.press_key(key)

        @tool
        async def open_app(package: str) -> str:
            """Launch an Android app by package name."""
            return await phone.open_app(package)

        return [read_screen, tap, type_text, swipe, press_key, open_app]

    async def run(self, goal: str, fallback_label: str) -> bool:
        t = self.p.t
        budget = min(self.MAX_UI_STEPS, t.remaining - 1)
        if budget <= 1:
            t.warn("not enough step budget left for UI agent")
            return False
        if self.p.llm:
            from langchain.agents import create_agent

            agent = create_agent(
                self.p.llm, self._tools(),
                system_prompt="You operate an Android phone through tools. Be minimal: never tap anything "
                              "not required by the goal. When the goal is achieved, reply DONE; if impossible, reply FAILED.")
            t.note(f"🤖 UI agent: {goal[:80]}… (≤{budget} tool calls)")
            try:
                # each ReAct turn = model node + tools node → ~2 graph steps per tool call
                res = await agent.ainvoke({"messages": [("user", goal)]}, {"recursion_limit": 2 * budget + 1})
                answer = str(res["messages"][-1].content)
                t.note(f"🤖 UI agent says: {answer[:80]}")
                if "DONE" in answer.upper():
                    return True
            except BudgetExceeded:
                raise
            except Exception as e:
                t.warn(f"UI agent stopped ({type(e).__name__}) → deterministic fallback")
        # deterministic fallback: find a button whose label matches and tap it
        screen = await self.p.phone.read_screen()
        for line in screen.splitlines():
            m = re.search(r"'([^']*)' @\((\d+),(\d+)\)", line)
            if m and fallback_label in m.group(1).lower():
                await self.p.phone.tap(int(m.group(2)), int(m.group(3)))
                return True
        t.warn(f"could not find a '{fallback_label}' button on screen")
        return False


# ── dedup across polling cycles ─────────────────────────────────────────────
class SeenState:
    def __init__(self, path):
        self.path = path
        try:
            self.seen: set[str] = set(json.loads(path.read_text()))
        except Exception:
            self.seen = set()

    @staticmethod
    def _key(i: ActionItem) -> str:
        return hashlib.sha1(f"{i.source}|{i.sender}|{i.summary}".encode()).hexdigest()[:16]

    def filter_new(self, d: Digest) -> Digest:
        return Digest(items=[i for i in d.items if self._key(i) not in self.seen], research_topics=d.research_topics)

    def mark(self, d: Digest) -> None:
        self.seen |= {self._key(i) for i in d.items}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.seen)))
