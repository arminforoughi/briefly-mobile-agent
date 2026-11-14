"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Default mapping from our normalized phone primitives -> Phone Use MCP tool names.
# Values in "{braces}" are substituted from the primitive's keyword arguments.
# Defaults target the open-source `mobile-mcp` server; override with PHONE_MCP_TOOLMAP.
DEFAULT_TOOLMAP: dict[str, dict] = {
    "read_screen": {"tool": "mobile_list_elements_on_screen", "args": {}},
    "tap": {"tool": "mobile_click_on_screen_at_coordinates", "args": {"x": "{x}", "y": "{y}"}},
    "type_text": {"tool": "mobile_type_keys", "args": {"text": "{text}", "submit": False}},
    "open_app": {"tool": "mobile_launch_app", "args": {"packageName": "{package}"}},
    "press_key": {"tool": "mobile_press_button", "args": {"button": "{key}"}},
    # Phone Use servers that expose these get used; otherwise we fall back to ADB.
    "read_notifications": {"tool": "phone_read_notifications", "args": {"limit": "{limit}"}},
    "read_sms": {"tool": "phone_read_sms", "args": {"limit": "{limit}"}},
    "swipe": {"tool": "phone_swipe", "args": {"x1": "{x1}", "y1": "{y1}", "x2": "{x2}", "y2": "{y2}"}},
}


@dataclass
class Config:
    # LLM (any LangChain `init_chat_model` string works, e.g. "openai:gpt-4.1")
    model: str = field(default_factory=lambda: _env("BRIEFING_MODEL", "anthropic:claude-sonnet-5-5"))

    # Control loop
    max_steps: int = field(default_factory=lambda: int(_env("MAX_STEPS", "50")))
    max_research_topics: int = field(default_factory=lambda: int(_env("MAX_RESEARCH_TOPICS", "3")))
    poll_interval_s: int = field(default_factory=lambda: int(_env("POLL_INTERVAL_S", "300")))

    # Phone
    phone_backend: str = field(default_factory=lambda: _env("PHONE_BACKEND", "mcp"))  # phoneuse | mcp | adb | mock
    adb_serial: str = field(default_factory=lambda: _env("ADB_SERIAL"))
    mcp_command: list[str] = field(
        default_factory=lambda: shlex.split(_env("PHONE_MCP_COMMAND", "npx -y @mobilenext/mobile-mcp@latest"))
    )
    mcp_url: str = field(default_factory=lambda: _env("PHONE_MCP_URL"))  # streamable-http alternative
    mcp_api_key: str = field(default_factory=lambda: _env("PHONE_MCP_API_KEY"))  # PhoneUse pbk_… key
    device_id: str = field(default_factory=lambda: _env("PHONE_DEVICE_ID"))
    mcp_toolmap: dict = field(
        default_factory=lambda: {**DEFAULT_TOOLMAP, **json.loads(_env("PHONE_MCP_TOOLMAP", "{}") or "{}")}
    )
    email_app: str = field(default_factory=lambda: _env("EMAIL_APP_PACKAGE", "com.google.android.gm"))
    sms_app: str = field(default_factory=lambda: _env("SMS_APP_PACKAGE", "com.google.android.apps.messaging"))

    # Research / voice
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))
    elevenlabs_voice_id: str = field(default_factory=lambda: _env("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"))
    kokoro_voice: str = field(default_factory=lambda: _env("KOKORO_VOICE", "af_heart"))
    elevenlabs_model: str = field(default_factory=lambda: _env("ELEVENLABS_MODEL", "eleven_multilingual_v2"))

    # Delivery
    deliver: list[str] = field(
        default_factory=lambda: [m.strip() for m in _env("DELIVER", "audio,sms").split(",") if m.strip()]
    )
    user_phone_number: str = field(default_factory=lambda: _env("USER_PHONE_NUMBER"))
    phone_audio_dir: str = field(default_factory=lambda: _env("PHONE_AUDIO_DIR", "/sdcard/Download"))
    twilio_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_from: str = field(default_factory=lambda: _env("TWILIO_FROM_NUMBER"))
    public_audio_base_url: str = field(default_factory=lambda: _env("PUBLIC_AUDIO_BASE_URL"))
    deliver_send: bool = field(default_factory=lambda: _env("DELIVER_SEND", "0") in ("1", "true", "yes"))
    dry_run: bool = field(default_factory=lambda: _env("DRY_RUN", "0") in ("1", "true", "yes"))

    # Output
    runs_dir: Path = field(default_factory=lambda: Path(_env("RUNS_DIR", "runs")))
    state_file: Path = field(default_factory=lambda: Path(_env("STATE_FILE", "runs/.seen.json")))

    @property
    def llm_enabled(self) -> bool:
        provider = self.model.split(":", 1)[0]
        key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
        return bool(os.getenv(key)) if key else True
