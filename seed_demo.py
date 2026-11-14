"""Seed the PhoneUse cloud phone with demo conversations for the judges.

The cloud phone has no carrier, so it can't receive real SMS. Instead this
creates one Messaging conversation per demo contact and leaves the text as a
draft. The conversation list then shows sender + message preview, which is
exactly what the briefing agent reads.

    .venv/bin/python seed_demo.py            # seed all
    .venv/bin/python seed_demo.py --only 1   # just the first (smoke test)
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from briefing_agent.config import Config
from briefing_agent.phoneuse import PhoneUseBackend

# 555-01xx numbers are reserved for fiction.
DEMO = [
    ("4155550142", "URGENT from Dana: client wants EU AI Act compliance talking points by 2pm today"),
    ("4155550177", "Priya: board sync moved to 3:30pm, Room Atlas. Please skim the Q4 deck before the Nvidia call"),
    ("4155550123", "Jordan: offsite moved to Dec 4, hotel confirmed. Need your headcount by Friday"),
    ("6505550199", "Mom: dinner Sunday still on?"),
]


async def find(phone: PhoneUseBackend, *labels: str):
    for text, x, y in await phone._ocr():
        if any(lbl in text.lower() for lbl in labels):
            return x, y, text
    return None


async def seed_one(phone: PhoneUseBackend, number: str, body: str) -> bool:
    for _ in range(3):                              # unwind any half-open compose screen
        await phone.press_key("back")
    await phone._open_sms_app()
    await asyncio.sleep(1.5)
    await phone.tap(893, 950)                       # "+" start-chat button
    await asyncio.sleep(1.5)
    await phone.type_text(number)
    await phone.press_key("enter")
    await asyncio.sleep(2.0)
    field = await find(phone, "type message", "text message", "message")
    if not field:
        print(f"  ✗ {number}: compose field not found")
        return False
    await phone.tap(field[0], field[1])
    await phone.type_text(body)
    await asyncio.sleep(0.8)
    await phone.tap(67, 58)                         # toolbar ← arrow saves the draft (hardware Back discards it)
    await asyncio.sleep(1.5)
    print(f"  ✓ {number}: {body[:50]}…")
    return True


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=len(DEMO))
    ap.add_argument("--start", type=int, default=0, help="skip the first N contacts")
    args = ap.parse_args()
    cfg = Config()
    phone = PhoneUseBackend(cfg)
    phone.shots_dir = Path("runs/seed/screens")
    await phone.start()
    try:
        for number, body in DEMO[args.start : args.only]:
            await seed_one(phone, number, body)
        await phone._open_sms_app()
        await asyncio.sleep(1.5)
        print("\nConversation list now reads:")
        for t, _, _ in await phone._ocr():
            print("   ", t)
        await phone.press_key("home")
    finally:
        await phone.close()


if __name__ == "__main__":
    asyncio.run(main())
