"""Live demo dashboard for the mobile briefing agent.

    .venv/bin/python demo_server.py          # → http://localhost:8777 (PORT env overrides)

Streams every agent event (steps, phone screens, triage, research, script,
audio) to the browser over Server-Sent Events. Mode "mock" uses canned phone
data with stage pacing; mode "live" drives the PhoneUse cloud Pixel.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from briefing_agent.config import Config
from briefing_agent.main import run_cycle

ROOT = Path(__file__).parent.resolve()
PORT = int(os.getenv("PORT", "8777"))
mimetypes.add_type("audio/mp4", ".m4a")


class Hub:
    """Fan-out of agent events to SSE clients, with replay of the current run."""

    def __init__(self):
        self.clients: list[queue.Queue] = []
        self.history: list[dict] = []
        self.lock = threading.Lock()
        self.running = False

    def publish(self, ev: dict) -> None:
        with self.lock:
            if ev.get("event") == "meta":
                self.history = []
            self.history.append(ev)
            for q in self.clients:
                q.put(ev)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            for ev in self.history:
                q.put(ev)
            self.clients.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


HUB = Hub()
LOOP = asyncio.new_event_loop()
threading.Thread(target=LOOP.run_forever, daemon=True).start()


def config_for(mode: str) -> tuple[Config, float]:
    cfg = Config()
    if mode == "live":
        cfg.phone_backend, cfg.deliver = "phoneuse", ["phoneuse"]
        return cfg, 0.0
    cfg.phone_backend, cfg.deliver = "mock", ["audio", "sms"]
    cfg.user_phone_number = cfg.user_phone_number or "+1 415 555 0100"
    return cfg, 0.9  # stage pacing so the audience can follow


async def _run(mode: str) -> None:
    cfg, pace = config_for(mode)
    try:
        await run_cycle(cfg, 1, None, force=True, listener=HUB.publish, pace=pace)
    except Exception as e:  # never kill the server mid-demo
        HUB.publish({"event": "warn", "msg": f"run crashed: {e}"})
        HUB.publish({"event": "done", "used": 0, "max": cfg.max_steps})
    finally:
        HUB.running = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console for agent traces
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, (ROOT / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/events":
            return self._sse()
        if path.startswith("/runs/"):
            f = (ROOT / path.lstrip("/")).resolve()
            if ROOT / "runs" in f.parents and f.is_file():
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/start"):
            mode = "live" if "mode=live" in self.path else "mock"
            if HUB.running:
                return self._send(409, b'{"error":"already running"}', "application/json")
            HUB.running = True
            asyncio.run_coroutine_threadsafe(_run(mode), LOOP)
            return self._send(202, json.dumps({"mode": mode}).encode(), "application/json")
        self._send(404, b"not found", "text/plain")

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = HUB.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                    data = json.dumps(ev, default=str)
                    self.wfile.write(f"data: {data}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            HUB.unsubscribe(q)


if __name__ == "__main__":
    print(f"Briefing demo → http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
