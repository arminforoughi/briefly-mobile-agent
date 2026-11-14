"""CLI entry point:  python -m briefing_agent [--mock] [--once] ..."""
from __future__ import annotations

import argparse
import asyncio
import signal
from datetime import datetime

from .config import Config
from .phone import TracedPhone, make_backend
from .pipeline import BriefingPipeline, SeenState, make_llm
from .tracer import BudgetExceeded, Tracer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="briefing_agent", description="Mobile AI briefing agent")
    p.add_argument("--mock", action="store_true", help="use the mock phone (no device needed)")
    p.add_argument("--backend", choices=["phoneuse", "mcp", "adb", "mock"], help="phone backend (default: env PHONE_BACKEND or mcp)")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--interval", type=int, help="seconds between polling cycles")
    p.add_argument("--max-steps", type=int, help="hard step budget per cycle (default 50)")
    p.add_argument("--deliver", help="comma list: audio,sms,call")
    p.add_argument("--dry-run", action="store_true", help="build the brief but don't act on the phone")
    p.add_argument("--force", action="store_true", help="brief even if nothing new since last cycle")
    return p.parse_args()


async def run_cycle(cfg: Config, cycle: int, state: SeenState | None, force: bool,
                    listener=None, pace: float = 0.0) -> None:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracer = Tracer(cfg.runs_dir / run_id, max_steps=cfg.max_steps, pace=pace)
    if listener:
        tracer.listeners.append(listener)
    tracer.emit("meta", run_id=run_id, backend=cfg.phone_backend, max_steps=cfg.max_steps,
                coord=[1000, 1000] if cfg.phone_backend == "phoneuse" else [1080, 2340],
                llm=cfg.model if cfg.llm_enabled else "heuristic", deliver=cfg.deliver)
    tracer.banner(
        f"📱 Mobile Briefing Agent · cycle {cycle}",
        f"phone={cfg.phone_backend}  llm={cfg.model if cfg.llm_enabled else 'heuristic (no key)'}  "
        f"budget={cfg.max_steps} steps  deliver={','.join(cfg.deliver)}{'  DRY-RUN' if cfg.dry_run else ''}",
    )
    llm = make_llm(cfg)
    backend = make_backend(cfg, llm)
    if hasattr(backend, "shots_dir"):
        backend.shots_dir = tracer.run_dir / "screens"
    backend.on_screen = lambda view: tracer.emit("screen", **view)
    result, extra = None, {}
    try:
        await tracer.step("PHONE", "connect", backend.start)
        if getattr(backend, "routed", None):
            tracer.note("routing: " + ", ".join(f"{k}→{v}" for k, v in backend.routed.items()))
        pipe = BriefingPipeline(cfg, TracedPhone(backend, tracer), tracer, llm)
        result = await pipe.run(state, force=force)
    except BudgetExceeded as e:
        tracer.warn(f"🛑 {e} — halting cycle safely")
    except Exception as e:
        tracer.warn(f"cycle aborted: {type(e).__name__}: {e}")
    finally:
        try:
            await backend.close()
        except Exception:
            pass
    if result:
        extra = {
            "action items": str(len(result.digest.items)),
            "research notes": str(len(result.research)),
            "audio": result.audio_path or "—",
            "delivered via": ", ".join(result.delivered) or "—",
        }
    tracer.summary(extra)
    tracer.emit("done", used=tracer.used, max=tracer.max_steps)
    tracer.close()


async def amain() -> None:
    args = parse_args()
    cfg = Config()
    if args.mock:
        cfg.phone_backend = "mock"
    if args.backend:
        cfg.phone_backend = args.backend
    if args.interval:
        cfg.poll_interval_s = args.interval
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.deliver:
        cfg.deliver = [m.strip() for m in args.deliver.split(",") if m.strip()]
    if args.dry_run:
        cfg.dry_run = True

    state = SeenState(cfg.state_file)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    cycle = 1
    while not stop.is_set():
        await run_cycle(cfg, cycle, state, force=args.force or args.once)
        if args.once:
            break
        cycle += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=cfg.poll_interval_s)
        except asyncio.TimeoutError:
            pass


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
