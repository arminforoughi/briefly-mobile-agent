"""Keyless local fallbacks: Apple Vision OCR, DuckDuckGo search, macOS `say` TTS."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

CHROME = re.compile(
    r"^(\d{1,2}:\d{2}.*|.*\d%|internet|bluetooth|flashlight|do not disturb|manage|silent|clear all|"
    r"verizon.*|messaging|.{0,2})$", re.I)


def ocr_elements(img: bytes) -> list[tuple[str, int, int]]:
    """On-device OCR (macOS Vision). Returns (text, x, y) with centers in the 0-1000 grid, top-to-bottom."""
    import Vision
    from Foundation import NSData

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(img, len(img)), None)
    handler.performRequests_error_([req], None)
    out = []
    for obs in req.results() or []:
        text = str(obs.topCandidates_(1)[0].string()).strip()
        bb = obs.boundingBox()  # normalized, origin bottom-left
        x = int((bb.origin.x + bb.size.width / 2) * 1000)
        y = int((1 - (bb.origin.y + bb.size.height / 2)) * 1000)
        if text:
            out.append((text, x, y))
    return sorted(out, key=lambda e: (e[2], e[1]))


def content_lines(els: list[tuple[str, int, int]]) -> list[str]:
    """OCR lines minus status-bar / quick-settings chrome."""
    lines = (t.lstrip("•*·- ").strip() for t, _, _ in els)
    return [t for t in lines if not CHROME.match(t)]


def web_search(query: str, n: int = 4) -> dict:
    import time

    from ddgs import DDGS

    results, last = [], None
    for backend in ("auto", "bing", "auto", "duckduckgo", "auto"):  # free engines are individually flaky
        try:
            results = DDGS().text(query, max_results=n, backend=backend) or []
            if results:
                break
        except Exception as e:
            last = e
            time.sleep(0.6)
    if not results:
        raise RuntimeError(f"all free search backends failed: {last}")
    return {"answer": "", "results": [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in results]}


def say_tts(script: str, out_base: Path, voice: str = "Samantha") -> Path:
    aiff, m4a = out_base.with_suffix(".aiff"), out_base.with_suffix(".m4a")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), script], check=True)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(aiff), str(m4a)], check=True)
    aiff.unlink(missing_ok=True)
    return m4a


_KOKORO = None
KOKORO_DIR = Path(__file__).resolve().parent.parent / "models"


def kokoro_available() -> bool:
    return (KOKORO_DIR / "kokoro-v1.0.int8.onnx").exists() and (KOKORO_DIR / "voices-v1.0.bin").exists()


def kokoro_tts(script: str, out_base: Path, voice: str = "af_heart", speed: float = 1.05) -> Path:
    """Open-source Kokoro-82M (ONNX, runs locally on CPU, no key)."""
    global _KOKORO
    import soundfile as sf
    from kokoro_onnx import Kokoro

    if _KOKORO is None:
        _KOKORO = Kokoro(str(KOKORO_DIR / "kokoro-v1.0.int8.onnx"), str(KOKORO_DIR / "voices-v1.0.bin"))
    samples, sr = _KOKORO.create(script, voice=voice, speed=speed, lang="en-us")
    wav, m4a = out_base.with_suffix(".wav"), out_base.with_suffix(".m4a")
    sf.write(str(wav), samples, sr)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(m4a)], check=True)
    wav.unlink(missing_ok=True)
    return m4a


def ocr_find(img: bytes, word: str) -> tuple[int, int] | None:
    """Locate a word inside OCR lines (Vision returns whole lines) → its center in the 0-1000 grid."""
    import Vision
    from Foundation import NSData

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(img, len(img)), None).performRequests_error_([req], None)
    for obs in req.results() or []:
        text = str(obs.topCandidates_(1)[0].string())
        i = text.lower().find(word.lower())
        if i < 0:
            continue
        bb = obs.boundingBox()
        frac = (i + len(word) / 2) / max(len(text), 1)
        return int((bb.origin.x + bb.size.width * frac) * 1000), int((1 - (bb.origin.y + bb.size.height / 2)) * 1000)
    return None
