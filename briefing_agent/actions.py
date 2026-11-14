"""Interactive follow-ups after the brief: reply to a message, order food.

Every side effect goes through a human confirmation gate: the agent drives the
phone up to the last screen (message typed / checkout open), then asks
`confirm(payload)` and only taps Send / Place order on an explicit yes.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from .config import Config
from .phone import TracedPhone, make_backend
from .tracer import BudgetExceeded, Tracer

ConfirmFn = Callable[[dict], Awaitable[bool]]

# demo address book (fictional 555 numbers, same ones seed_demo.py creates)
CONTACTS = {
    "dana": ("Dana", "4155550142"),
    "priya": ("Priya", "4155550177"),
    "jordan": ("Jordan", "4155550123"),
    "mom": ("Mom", "6505550199"),
}

DOORDASH_PKG = "com.dd.doordash"

# mock DoorDash catalogue — clearly fake prices, no real ordering possible
MENU = {
    "Thai Basil": {"eta": "25–35 min", "rating": "4.7", "items": {
        "Pad Thai": 16.50, "Green Curry": 17.00, "Spring Rolls": 7.50, "Thai Iced Tea": 5.00}},
    "Sweetgreen": {"eta": "15–25 min", "rating": "4.6", "items": {
        "Harvest Bowl": 15.95, "Kale Caesar": 13.45, "Chicken Pesto Parm": 16.25}},
    "Chipotle": {"eta": "20–30 min", "rating": "4.4", "items": {
        "Burrito Bowl": 12.85, "Chips & Guac": 5.95, "Chicken Burrito": 12.40}},
    "Tony's Pizza": {"eta": "30–40 min", "rating": "4.8", "items": {
        "Margherita Pizza": 18.00, "Pepperoni Pizza": 20.00, "Caesar Salad": 11.00}},
}
STORE_HINTS = {"thai": "Thai Basil", "pad": "Thai Basil", "curry": "Thai Basil", "salad": "Sweetgreen",
               "harvest": "Sweetgreen", "burrito": "Chipotle", "taco": "Chipotle", "guac": "Chipotle",
               "pizza": "Tony's Pizza", "sweetgreen": "Sweetgreen", "chipotle": "Chipotle"}


class Intent(BaseModel):
    kind: Literal["send_message", "order_food", "unknown"]
    recipient: str | None = Field(default=None, description="contact name or phone number")
    body: str | None = Field(default=None, description="message text to send")
    store: str | None = Field(default=None, description="restaurant name")
    items: list[str] = Field(default_factory=list, description="dishes to order")

    def summary_line(self) -> str:
        if self.kind == "send_message":
            return f"send_message → {self.recipient}: {self.body!r}"
        if self.kind == "order_food":
            return f"order_food → {', '.join(self.items) or 'anything'} @ {self.store or 'auto'}"
        return "unknown"


# ── intent parsing ──────────────────────────────────────────────────────────
def parse_rules(text: str) -> Intent:
    t = text.strip()
    low = t.lower()
    if re.search(r"\b(order|doordash|get me|hungry|lunch|dinner)\b", low):
        store = None
        m = re.search(r"\bfrom ([\w' &]+?)(?:\s+(?:on|via|using) doordash|[.,!]|$)", low)
        if m and "doordash" not in m.group(1):
            store = m.group(1).strip()
        items: list[str] = []
        im = re.search(r"\b(?:order|get me|get)\s+(?:me\s+)?(?:some\s+|a\s+|an\s+)?(.+?)(?:\s+from\b|\s+(?:on|via|using) doordash|$)", low)
        if im:
            items = [x.strip() for x in re.split(r",|\band\b", im.group(1))
                     if x.strip() and x.strip() not in ("lunch", "dinner", "food", "something", "doordash")]
        return Intent(kind="order_food", store=store, items=items)
    m = re.match(r"^(?:reply|respond|text|message|tell|send)\s+(?:back\s+)?(?:to\s+)?(.+)$", t, re.I | re.S)
    if m:
        rest = m.group(1)
        num = re.match(r"^(\+?[\d\s()\-]{7,})[:,]?\s*(.*)$", rest, re.S)
        if num:
            who, body = num.group(1).strip(), num.group(2)
        else:
            for key, (name, _) in CONTACTS.items():
                if rest.lower().startswith(key):
                    who, body = name, rest[len(key):]
                    break
            else:
                parts = rest.split(None, 1)
                who, body = parts[0].strip(":,"), parts[1] if len(parts) > 1 else ""
        body = re.sub(r"^[\s:,\-–—]*(?:that\s+)?", "", body).strip()
        return Intent(kind="send_message", recipient=who, body=body or None)
    return Intent(kind="unknown")


def resolve_contact(who: str | None) -> tuple[str, str | None]:
    if not who:
        return "?", None
    digits = re.sub(r"\D", "", who)
    if len(digits) >= 7:
        for name, num in CONTACTS.values():
            if num.endswith(digits[-10:]):
                return name, num
        return who, digits
    c = CONTACTS.get(who.lower().strip())
    return (c[0], c[1]) if c else (who, None)


def fmt_number(n: str) -> str:
    d = re.sub(r"\D", "", n)[-10:]
    return f"({d[:3]}) {d[3:6]}-{d[6:]}" if len(d) == 10 else n


def suggest(items: list[dict]) -> list[str]:
    """One-tap follow-ups derived from the triage."""
    out: list[str] = []
    for it in items:
        blob = f"{it.get('sender', '')} {it.get('summary', '')}"
        for key, (name, _) in CONTACTS.items():
            if re.search(rf"\b{key}\b", blob, re.I):
                if it.get("priority") == "high":
                    m = re.search(r"\bby (\d{1,2}(?::\d\d)?\s?(?:am|pm))", blob, re.I)
                    out.append(f"Reply to {name}: On it, you'll have it {'before ' + m.group(1) if m else 'shortly'}")
                else:
                    out.append(f"Reply to {name}: Got it, thanks!")
                break
    out = list(dict.fromkeys(out))[:3]
    out.append("Order pad thai from Thai Basil on DoorDash")
    return out


# ── cart maths (mock) ───────────────────────────────────────────────────────
@dataclass
class Cart:
    store: str
    lines: list[tuple[str, int, float]]
    eta: str

    @property
    def subtotal(self) -> float:
        return round(sum(q * p for _, q, p in self.lines), 2)

    @property
    def fees(self) -> list[tuple[str, float]]:
        s = self.subtotal
        return [("Delivery fee", 2.99), ("Service fee", round(s * 0.10, 2)), ("Estimated tax", round(s * 0.0863, 2))]

    @property
    def total(self) -> float:
        return round(self.subtotal + sum(v for _, v in self.fees), 2)


def build_cart(intent: Intent) -> Cart:
    words = " ".join([intent.store or ""] + intent.items).lower()
    store = next((s for s in MENU if intent.store and intent.store.lower() in s.lower()), None)
    if not store:
        store = next((v for k, v in STORE_HINTS.items() if k in words), "Sweetgreen")
    menu = MENU[store]["items"]
    lines = []
    for want in intent.items:
        ww = set(re.findall(r"\w+", want.lower()))
        best = max(menu, key=lambda dish: len(ww & set(re.findall(r"\w+", dish.lower()))))
        if ww & set(re.findall(r"\w+", best.lower())):
            lines.append((best, 1, menu[best]))
    if not lines:
        first = next(iter(menu))
        lines.append((first, 1, menu[first]))
    return Cart(store, lines, MENU[store]["eta"])


# ── runner ──────────────────────────────────────────────────────────────────
class ActionRunner:
    def __init__(self, cfg: Config, phone: TracedPhone, backend, tracer: Tracer, llm, confirm: ConfirmFn):
        self.cfg, self.phone, self.backend, self.t, self.llm, self.confirm = cfg, phone, backend, tracer, llm, confirm
        self.mock = backend.name == "mock"

    def _show(self, **view):
        if self.backend.on_screen:
            self.backend.on_screen(view)

    async def _parse(self, text: str) -> Intent:
        if self.llm:
            try:
                return await self.llm.with_structured_output(Intent).ainvoke(
                    "Turn this phone command into an intent. Known contacts: "
                    + ", ".join(n for n, _ in CONTACTS.values()) + f"\n\nCommand: {text}")
            except Exception as e:
                self.t.warn(f"LLM intent parse failed → rules ({e})")
        return parse_rules(text)

    async def run(self, text: str) -> str:
        self.t.phase(f"ACTION · {text}")
        intent = await self.t.step("LLM", "parse_intent", self._parse, text)
        if intent.kind == "send_message":
            return await self.send_message(intent)
        if intent.kind == "order_food":
            return await self.order_food(intent)
        self.t.warn("I can reply to messages or order food — try “Reply to Dana: on it”.")
        return "Sorry, I can reply to messages or order food for now."

    # ── messaging ───────────────────────────────────────────────────────
    async def send_message(self, intent: Intent) -> str:
        name, number = resolve_contact(intent.recipient)
        if not number:
            self.t.warn(f"I don't have a number for {name}")
            return f"I don't have a number for {name}."
        body = intent.body or "On it!"
        if self.mock:
            history = [s.body for s in self.backend._sms() if name.lower() in s.sender.lower()][:1]
            await self.phone.open_app(self.cfg.sms_app)
            self._show(view="thread", name=name, number=fmt_number(number), history=history, draft="", sent=False)
            await self.phone.tap(480, 2240)
            await self.phone.type_text(body)
            self._show(view="thread", name=name, number=fmt_number(number), history=history, draft=body, sent=False)
        else:
            await self._live_compose(number, body)

        ok = await self.confirm({
            "kind": "message", "title": f"Send this to {name}?", "cta": "Send",
            "lines": [["To", f"{name} · {fmt_number(number)}"], ["Message", body]],
        })
        if not ok:
            if self.mock:
                self._show(view="thread", name=name, number=fmt_number(number), history=history, draft="", sent=False)
            else:
                await self.phone.tap(67, 58)  # ← saves it as a draft
            self.t.note("✋ cancelled by you — nothing sent")
            return f"Okay, I didn't send it. The message to {name} is saved as a draft." if not self.mock else "Okay, I didn't send it."
        if self.mock:
            await self.phone.tap(990, 2240)
            self._show(view="thread", name=name, number=fmt_number(number), history=history, draft=body, sent=True)
        else:
            await self.phone.tap(919, 956)  # send arrow (0-1000 grid)
            await asyncio.sleep(1.5)
            await self.phone.read_screen()
            await self.phone.tap(67, 58)
        self.t.note(f"✅ sent to {name}")
        return f"Done. I told {name}: {body}"

    async def _live_compose(self, number: str, body: str) -> None:
        await self.phone.compose_sms(number, body)  # PhoneUse: opens Messaging
        await self.phone.tap(893, 950)              # + start chat
        await asyncio.sleep(1.5)
        await self.phone.type_text(number)
        await self.phone.press_key("enter")
        await asyncio.sleep(2.0)
        screen = await self.phone.read_screen()
        m = re.search(r"'(?:Send message|Text message)[^']*' @\((\d+),(\d+)\)", screen, re.I)
        if m:
            await self.phone.tap(int(m.group(1)), int(m.group(2)))
        else:  # the conversation already has a draft in the box → clear it so we don't splice into it
            await self.phone.clear_field(300, 958)
            await self.phone.tap(300, 958)
        await self.phone.type_text(body)

    # ── food ────────────────────────────────────────────────────────────
    async def order_food(self, intent: Intent) -> str:
        if not self.mock:
            return await self._live_order(intent)
        cart = await self.t.step("LLM", "build_cart", build_cart, intent)
        store = MENU[cart.store]
        await self.phone.open_app(DOORDASH_PKG)
        self._show(view="dd_home", query="")
        await self.phone.tap(540, 330)
        await self.phone.type_text(cart.store)
        self._show(view="dd_home", query=cart.store, results=[
            {"name": s, "eta": m["eta"], "rating": m["rating"], "hit": s == cart.store}
            for s, m in sorted(MENU.items(), key=lambda kv: kv[0] != cart.store)][:3])
        await self.phone.tap(540, 620)
        added: list[str] = []
        menu = [{"name": k, "price": v} for k, v in store["items"].items()]
        self._show(view="dd_menu", store=cart.store, eta=store["eta"], rating=store["rating"], items=menu, added=added)
        for dish, _, _ in cart.lines:
            idx = list(store["items"]).index(dish)
            await self.phone.tap(960, 900 + idx * 260)
            added.append(dish)
            self._show(view="dd_menu", store=cart.store, eta=store["eta"], rating=store["rating"], items=menu, added=list(added))
        await self.phone.tap(540, 2200)
        cart_view = dict(view="dd_cart", store=cart.store, lines=[[d, q, p] for d, q, p in cart.lines],
                         fees=cart.fees, subtotal=cart.subtotal, total=cart.total, eta=cart.eta)
        self._show(**cart_view)
        ok = await self.confirm({
            "kind": "order", "title": f"Place order at {cart.store}?", "cta": f"Place order · ${cart.total:.2f}",
            "lines": [[f"{q}× {d}", f"${p * q:.2f}"] for d, q, p in cart.lines]
                     + [[k, f"${v:.2f}"] for k, v in cart.fees],
            "total": f"${cart.total:.2f}", "note": f"Arrives in {cart.eta} · mock order, no real charge",
        })
        if not ok:
            await self.phone.press_key("back")
            self.t.note("✋ cancelled by you — nothing ordered")
            return "No problem, I cancelled the order. Nothing was charged."
        await self.phone.tap(540, 2200)
        self._show(view="dd_placed", store=cart.store, eta=cart.eta, total=f"${cart.total:.2f}",
                   items=[d for d, _, _ in cart.lines])
        self.t.note(f"✅ order placed at {cart.store} · ${cart.total:.2f}")
        dishes = " and ".join(d for d, _, _ in cart.lines)
        return f"Ordered. {dishes} from {cart.store} is on the way, about {cart.eta.replace('–', ' to ')}."

    async def _live_order(self, intent: Intent) -> str:
        try:
            await self.phone.open_app(DOORDASH_PKG)
        except BudgetExceeded:
            raise
        except Exception:
            self.t.warn("DoorDash isn't installed on this PhoneUse device — install & sign in via the PhoneUse "
                        "console, or switch to Mock mode for the demo.")
            return "DoorDash isn't installed on this phone yet, so I couldn't order. Try Mock mode for the demo."
        what = ", ".join(intent.items) or "a popular lunch"
        where = f" from {intent.store}" if intent.store else ""
        res = await self.phone.agent_run(
            f"In the DoorDash app, search for{where or ' a nearby restaurant'}, add {what} to the cart and open "
            "checkout. STOP on the checkout screen. Do NOT tap Place Order.", max_steps=30, timeout_s=300)
        if "succeeded" not in res:
            return f"I couldn't get to checkout ({res})."
        screen = await self.phone.read_screen()
        money = [ln.split("'")[1] for ln in screen.splitlines() if "$" in ln and "'" in ln][-6:]
        ok = await self.confirm({"kind": "order", "title": f"Place this DoorDash order{where}?",
                                 "lines": [[m, ""] for m in money] or [["Checkout is open on the phone", ""]],
                                 "cta": "Place order", "note": "Charged to the card saved in DoorDash"})
        if not ok:
            return "Cancelled. Nothing was ordered."
        res = await self.phone.agent_run("On the DoorDash checkout screen, tap Place Order once and wait for the "
                                         "order confirmation.", max_steps=10, timeout_s=120)
        return "Order placed on DoorDash." if "succeeded" in res else f"Placing the order didn't confirm ({res})."


# ── one-shot entry point (used by the demo server) ──────────────────────────
async def run_action(cfg: Config, text: str, listener, confirm: ConfirmFn, pace: float = 0.0) -> str | None:
    from .pipeline import BriefingPipeline, make_llm

    run_id = "act-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tracer = Tracer(cfg.runs_dir / run_id, max_steps=cfg.max_steps, pace=pace)
    tracer.listeners.append(listener)
    tracer.emit("action_start", run_id=run_id, text=text, backend=cfg.phone_backend, max_steps=cfg.max_steps,
                coord=[1000, 1000] if cfg.phone_backend == "phoneuse" else [1080, 2340])
    llm = make_llm(cfg)
    backend = make_backend(cfg, llm)
    if hasattr(backend, "shots_dir"):
        backend.shots_dir = tracer.run_dir / "screens"
    backend.on_screen = lambda view: tracer.emit("screen", **view)
    msg = None
    phone = TracedPhone(backend, tracer)
    try:
        await tracer.step("PHONE", "connect", backend.start)
        msg = await ActionRunner(cfg, phone, backend, tracer, llm, confirm).run(text)
    except BudgetExceeded as e:
        tracer.warn(f"🛑 {e}")
        msg = "I hit my step budget, so I stopped."
    except Exception as e:
        tracer.warn(f"action failed: {type(e).__name__}: {e}")
        msg = "Something went wrong on the phone, so I stopped."
    finally:
        try:
            await backend.close()
        except Exception:
            pass
    if msg:
        url = None
        try:
            tracer.pace = 0
            out = await BriefingPipeline(cfg, phone, tracer, llm).synthesize(msg)
            url = "/" + out if out else None
        except Exception:
            pass
        tracer.emit("say", text=msg, url=url)
    tracer.emit("action_done", used=tracer.used, max=tracer.max_steps)
    tracer.close()
    return msg
