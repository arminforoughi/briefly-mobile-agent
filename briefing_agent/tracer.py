"""Step budget + judge-friendly tracing.

Every phone primitive, LLM call, web search and TTS call goes through
`Tracer.step(...)`, which:
  * consumes one unit of the global step budget (hard cap, default 50),
  * times the call,
  * prints a single aligned trace line to the console,
  * appends a structured record to runs/<id>/trace.jsonl.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from rich.markup import escape as _esc

    _console: Console | None = Console(highlight=False)
except ImportError:
    _console = None

    def _esc(s: str) -> str:
        return s

KIND_STYLE = {
    "PHONE": "cyan",
    "LLM": "magenta",
    "WEB": "green",
    "TTS": "yellow",
    "DELIVER": "bold blue",
    "AGENT": "bright_magenta",
}


class BudgetExceeded(RuntimeError):
    pass


def _summarize(result: Any, width: int = 70) -> str:
    if result is None:
        return "ok"
    if isinstance(result, (list, tuple)):
        return f"{len(result)} item(s)"
    if hasattr(result, "summary_line"):
        return result.summary_line()
    text = str(result).replace("\n", " ⏎ ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _fmt_args(args: tuple, kwargs: dict, width: int = 40) -> str:
    parts = [repr(a) if not isinstance(a, str) else a for a in args]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    s = " ".join(p.replace("\n", " ") for p in parts)
    return s if len(s) <= width else s[: width - 1] + "…"


@dataclass
class Tracer:
    run_dir: Path
    max_steps: int = 50
    used: int = 0
    records: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)
    listeners: list = field(default_factory=list)  # live UIs subscribe here
    pace: float = 0.0  # demo pacing: seconds to wait before each step

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "trace.jsonl").open("a", encoding="utf-8")

    # ── budget ──────────────────────────────────────────────────────────
    @property
    def remaining(self) -> int:
        return self.max_steps - self.used

    def _consume(self) -> int:
        if self.used >= self.max_steps:
            raise BudgetExceeded(f"step budget of {self.max_steps} exhausted")
        self.used += 1
        return self.used

    # ── output helpers ──────────────────────────────────────────────────
    def _print(self, msg: str, style: str | None = None) -> None:
        if _console:
            _console.print(msg, style=style)
        else:
            print(msg)

    def _write(self, rec: dict) -> None:
        rec["t"] = round(time.perf_counter() - self.t0, 3)
        self.records.append(rec)
        self._jsonl.write(json.dumps(rec, default=str) + "\n")
        self._jsonl.flush()
        self._notify(rec)

    def _notify(self, rec: dict) -> None:
        for fn in self.listeners:
            try:
                fn(rec)
            except Exception:
                pass

    def emit(self, event: str, **data) -> None:
        """Structured event for live UIs (also persisted to trace.jsonl)."""
        self._write({"event": event, **data})

    def banner(self, title: str, subtitle: str = "") -> None:
        if _console:
            _console.print(Panel.fit(f"[bold]{title}[/bold]\n{subtitle}", border_style="bright_blue"))
        else:
            print(f"==== {title} ====\n{subtitle}")
        self._write({"event": "banner", "title": title, "subtitle": subtitle})

    def phase(self, name: str) -> None:
        self._print(f"\n[bold white on blue] {name} [/]" if _console else f"\n### {name}")
        self._write({"event": "phase", "name": name})

    def note(self, msg: str) -> None:
        self._print(f"   [dim]│[/dim] {_esc(msg)}" if _console else f"   | {msg}")
        self._write({"event": "note", "msg": msg})

    def warn(self, msg: str) -> None:
        self._print(f"   [yellow]⚠ {_esc(msg)}[/yellow]" if _console else f"   ! {msg}")
        self._write({"event": "warn", "msg": msg})

    # ── the core traced step ────────────────────────────────────────────
    async def step(self, kind: str, name: str, fn: Callable[..., Any | Awaitable[Any]], *args, **kwargs) -> Any:
        n = self._consume()
        self._notify({"event": "step_start", "n": n, "kind": kind, "name": name,
                      "args": _fmt_args(args, kwargs, 400), "max": self.max_steps})
        if self.pace:
            import asyncio

            await asyncio.sleep(self.pace)
        start = time.perf_counter()
        status, result, err = "ok", None, None
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 — trace then re-raise
            status, err = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            ms = int((time.perf_counter() - start) * 1000)
            style = KIND_STYLE.get(kind, "white")
            out = err if err else _summarize(result)
            mark = "[red]✗[/red]" if err else "[green]✓[/green]"
            argstr = _fmt_args(args, kwargs)
            if _console:
                self._print(
                    f" {mark} [dim]{n:02d}/{self.max_steps}[/dim]  [{style}]{kind:<7}[/{style}] "
                    f"[bold]{name:<20}[/bold] [dim]{_esc(argstr.ljust(40))}[/dim] → {_esc(out)} [dim]({ms} ms)[/dim]"
                )
            else:
                print(f" {'x' if err else '+'} {n:02d}/{self.max_steps} {kind:<7} {name:<20} {argstr:<40} -> {out} ({ms} ms)")
            self._write(
                {"event": "step", "n": n, "kind": kind, "name": name, "args": argstr, "max": self.max_steps,
                 "status": status, "result": out, "ms": ms}
            )

    def summary(self, extra: dict[str, str] | None = None) -> None:
        steps = [r for r in self.records if r.get("event") == "step"]
        by_kind: dict[str, int] = {}
        for r in steps:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        errors = sum(1 for r in steps if r["status"] == "error")
        elapsed = time.perf_counter() - self.t0
        rows = {
            "steps used": f"{self.used} / {self.max_steps}",
            "by kind": ", ".join(f"{k}={v}" for k, v in by_kind.items()),
            "errors (recovered)": str(errors),
            "wall time": f"{elapsed:.1f}s",
            "trace": str(self.run_dir / "trace.jsonl"),
            **(extra or {}),
        }
        if _console:
            t = Table(title="Run summary", show_header=False, border_style="bright_blue")
            for k, v in rows.items():
                t.add_row(f"[bold]{k}[/bold]", v)
            _console.print(t)
        else:
            for k, v in rows.items():
                print(f"{k:>20}: {v}")
        self._write({"event": "summary", **rows})

    def close(self) -> None:
        self._jsonl.close()
