# 📱 Mobile Briefing Agent

A Python agent built on LangChain and a Phone Use MCP server. It watches an Android phone,
triages what matters, researches it with Tavily and speaks a briefing with ElevenLabs.
Then it delivers that briefing back to the phone while you're away from the screen.

```
 ┌──────────── PHASE 1 · MONITOR & EXTRACT ────────────┐
 │ read_notifications · read_sms · open_app(Gmail) →   │
 │ read_screen · press_key(HOME) → LLM → Digest        │──┐
 └─────────────────────────────────────────────────────┘  │
 ┌──────────── PHASE 2 · DEEP RESEARCH ────────────────┐  │
 │ Tavily (advanced, parallel, ≤3 topics) → LLM bullets│◀─┘
 └─────────────────────────────────────────────────────┘──┐
 ┌──────────── PHASE 3 · VOICE BRIEFING ───────────────┐  │
 │ LLM script (60–90 s, spoken style) → ElevenLabs mp3 │◀─┘
 └─────────────────────────────────────────────────────┘──┐
 ┌──────────── PHASE 4 · PHONE DELIVERY ───────────────┐  │
 │ audio: push mp3 → open in native player (voice drop)│◀─┘
 │ sms:   compose intent → bounded UI agent taps Send  │
 │ call:  Twilio voice call (plays mp3 or <Say>)       │
 └─────────────────────────────────────────────────────┘
```

## Quick start

```bash
uv venv && uv pip install -r requirements.txt
./scripts/get_models.sh     # Kokoro voice model (~120 MB, optional)
cp .env.example .env        # fill in keys
```

Rehearse with no phone and no keys. This is the full pipeline with heuristic fallbacks:

```bash
.venv/bin/python -m briefing_agent --mock --once
```

Live demo with a real phone over ADB, a Phone Use MCP server and every API key:

```bash
.venv/bin/python -m briefing_agent --once
```

Run continuously. It polls every 5 minutes and only briefs when something new arrives:

```bash
.venv/bin/python -m briefing_agent --interval 300
```

| Flag | Meaning |
|---|---|
| `--mock` / `--backend {mcp,adb,mock}` | which phone layer to use |
| `--once` | one cycle (always briefs) |
| `--max-steps N` | hard step cap per cycle (default **50**) |
| `--deliver audio,sms,call` | delivery channels |
| `--dry-run` | build the brief, don't touch the phone |
| `--force` | brief even if nothing is new |

## 🎬 Demo dashboard (for judges)

```bash
.venv/bin/python demo_server.py        # → http://localhost:8777
```

The dashboard is a live view of the agent, streamed over Server-Sent Events:
* **Phone mirror:** each screen the agent visits, with a ripple wherever it taps. Mock mode draws the screens in HTML.
  Live mode shows real screenshots from the PhoneUse Pixel.
* **Pipeline and budget:** the four phases light up in turn, and a ring shows steps used out of 50.
* **Agent trace:** every step streams in with a colour-coded badge, its result and its timing.
* **Briefing:** priority-sorted triage cards, research bullets, and the voice brief with each word highlighted as it plays.

There are two modes, selected with the toggle in the header:
* **Mock phone** is offline-safe and paced for an audience. A full run takes about 17 s and 17 of 50 steps.
  The brief arrives as a voice drop played on the phone, and as an SMS.
* **Live Pixel · PhoneUse** drives the real cloud Pixel. It reads the screens with on-device OCR, and PhoneUse's own
  hosted agent saves the brief as a Messaging draft. A run takes about 2.5 min and 8 of 50 steps.
  It saves a draft only; set `DELIVER_SEND=1` to actually send.

### 💬 Talk back: reply and order, with a human in the loop

Under the voice brief, the **Ask Briefly** panel takes typed or spoken commands. The 🎤 button uses Chrome's speech
recognition. One-tap suggestions come from the triage, for example *"Reply to Dana: On it, you'll have it before 2pm"*.

* **Reply / text:** for example "Reply to Dana: on it" or "tell Mom Sunday works". The agent opens the thread on the phone
  and types the reply. If the box already holds a draft, it clears it first: long-press, Select all, delete.
* **Order food:** for example "order pad thai from Thai Basil on DoorDash" or "get me a burrito bowl". The agent searches,
  adds the dishes to the cart and opens checkout.
* **The confirmation gate:** the agent stops at the last screen and shows a card, such as *Send this to Dana?* or
  *Place order · $22.56*. Nothing is sent or bought until you press Confirm. Cancel backs out and saves the message
  as a draft, or leaves the order unplaced. Afterwards Briefly speaks the result in the Kokoro voice.

Mock mode uses a simulated DoorDash with fake prices, so no real charge is possible. In live mode, replies run on the
Pixel. DoorDash only works live once the app is installed and signed in on the PhoneUse device; until then the agent
says so.
The code is in [`briefing_agent/actions.py`](briefing_agent/actions.py).

To seed the live phone with demo conversations, run this once:

```bash
.venv/bin/python seed_demo.py
```

No API keys are required. Each missing key falls back to a local or free option:

| Missing key | Fallback |
|---|---|
| Anthropic | heuristic triage and a template script |
| Tavily | DuckDuckGo search (canned notes in mock mode) |
| ElevenLabs | the open-source Kokoro-82M voice, run locally (`KOKORO_VOICE`, default `af_heart`), then macOS `say` |
| Vision LLM | Apple Vision OCR |

## Phone Use primitives

All logic goes through one normalized interface in [`phone.py`](briefing_agent/phone.py):

`read_notifications · read_sms · read_screen · tap · type_text · swipe · open_app · press_key · push_file · open_media · compose_sms`

* **`MCPPhoneBackend`** opens a *persistent* MCP session through `langchain-mcp-adapters` and maps each
  primitive to a server tool via `PHONE_MCP_TOOLMAP`. The default map targets `mobile-mcp`.
  Any primitive the server doesn't expose falls back to ADB. Common examples are SMS and notification
  reads and file push. The routing table is printed at startup so judges can see which layer ran each step.
* **`ADBPhoneBackend`** talks to the device directly. It uses `dumpsys notification`, the `content://sms/inbox`
  query, `uiautomator dump` and `input tap/text/swipe/keyevent`, plus `am start` intents.
* **`PhoneUseBackend`** ([phoneuse.py](briefing_agent/phoneuse.py)) targets the hosted PhoneUse MCP. It leases the
  device and wakes it, then uses the 0–1000 tap grid. Cloud phones have no UI tree, so screens are read from
  screenshots by a vision LLM. Notifications and SMS are read through the UI: it pulls down the shade or opens
  Messaging. Every screenshot is saved to `runs/<id>/screens/`. Set `PHONE_BACKEND=phoneuse`, `PHONE_MCP_URL`,
  `PHONE_MCP_API_KEY` and `PHONE_DEVICE_ID`.
* **`MockPhoneBackend`** serves realistic canned data for rehearsals.

If your Phone Use server uses different tool names, remap them without code changes:

```bash
PHONE_MCP_TOOLMAP='{"read_sms":{"tool":"sms_list","args":{"max":"{limit}"}},"tap":{"tool":"click","args":{"x":"{x}","y":"{y}"}}}'
```

## Why it stays within 50 steps

* **Deterministic outer pipeline.** The LLM handles judgement: triage, research synthesis, script writing
  and UI navigation. It never decides control flow, so there are no runaway loops.
* **One shared hard budget.** Every phone primitive, LLM call, Tavily search and TTS call goes through
  `Tracer.step()`, and each costs one step. When the budget runs out, `BudgetExceeded` halts the cycle cleanly.
* **Bounded UI sub-agent.** The only free-form ReAct agent (`create_agent` over the phone tools) gets
  `recursion_limit = 2 × min(8, remaining)`. If it fails, a deterministic fallback finds the *Send* button
  in the screen dump and taps it.
* **Graceful degradation.** Missing keys or failing services make the agent fall back to heuristic triage,
  a template script or a skipped stage. It never crashes.

A typical live cycle uses about 15–20 of its 50 steps.

## Logging traces for the judges

Every step prints one aligned line:

```
 ✓ 02/50  PHONE   read_notifications   limit=15                    → 5 item(s) (50 ms)
 ✓ 07/50  LLM     extract_digest       Notifications …             → 6 items (3 high) · 3 research topics (2140 ms)
 ✓ 09/50  WEB     tavily_search        {'query': 'EU AI Act GPAI…  → {…} (1830 ms)
 ✓ 14/50  TTS     elevenlabs_tts       …                           → runs/…/brief.mp3 (212 KB) (2950 ms)
```

Each run also writes these files to `runs/<timestamp>/`:
* `trace.jsonl` holds the machine-readable steps (`n, kind, name, args, status, result, ms, t`) and is easy to replay or chart.
* `brief.txt` holds the spoken script.
* `brief.mp3` holds the ElevenLabs audio.

## Files

```
briefing_agent/
  config.py     env config + default MCP tool map
  phone.py      Phone Use backends (MCP / ADB / mock) + parsers + TracedPhone proxy
  tracer.py     step budget + rich console traces + JSONL
  pipeline.py   monitor → research → script → TTS → deliver, bounded UI agent, dedup state
  main.py       CLI + polling loop
```

## Notes

* The agent reads notification and SMS content in full (`dumpsys --noredact`, the SMS content provider).
  This needs USB debugging on a device you own.
* SMS delivery only goes to `USER_PHONE_NUMBER`, which should be your own number. Use `--dry-run` while rehearsing.
* The default model is `anthropic:claude-sonnet-5-5`. Change `BRIEFING_MODEL` to any provider LangChain supports.
