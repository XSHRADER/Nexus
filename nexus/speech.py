"""
speech.py
Voice input: turn a recording into the text of a question.

Two routes, chosen by the same cloud switch as everything else:
  * Cloud (switch on, Groq key set): Groq's hosted Whisper. Fast, and needs
    nothing installed.
  * Local (switch off, or the cloud call failed): `faster-whisper`, if it is
    installed (`pip install faster-whisper`). Runs on this PC.

With neither available, the error says exactly which one to set up.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from nexus import cloud, cloud_client
from nexus.config import get_settings


def local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _local_model(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="auto", compute_type="auto")


def _transcribe_local(audio: bytes, filename: str) -> str:
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(audio)
        path = fh.name
    try:
        segments, _info = _local_model(get_settings().local_whisper_model).transcribe(path)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        Path(path).unlink(missing_ok=True)


def transcribe(
    audio: bytes,
    filename: str = "recording.wav",
    mime: str = "audio/wav",
    policy: cloud.CloudPolicy | None = None,
) -> dict[str, Any]:
    """Returns {"text", "provider", "model", "local"}; raises RuntimeError with a
    fix-it message when no route is available."""
    if not audio:
        raise RuntimeError("The recording is empty.")
    settings = get_settings()
    policy = policy or cloud.CloudPolicy.from_settings()
    provider, model = settings.speech_provider, settings.speech_model

    errors: list[str] = []
    if policy.mode != "off":
        status = cloud.provider_status(provider)
        if status["status"] == "ready":
            try:
                text = cloud_client.transcribe(provider, model, audio, filename, mime)
                cloud.record_success(provider, len(audio), len(text))
                return {"text": text, "provider": provider, "model": model, "local": False}
            except cloud.CloudError as exc:
                cloud.record_failure(exc, model)
                errors.append(str(exc))
        else:
            errors.append(f"{provider}: {status['detail']}")

    if local_available():
        text = _transcribe_local(audio, filename)
        if not text:
            raise RuntimeError("No speech was recognised in the recording.")
        return {"text": text, "provider": "faster-whisper",
                "model": settings.local_whisper_model, "local": True}

    if policy.mode == "off":
        raise RuntimeError(
            "Voice input needs a speech model. Either install one locally "
            "(`pip install faster-whisper`) or switch cloud on and set GROQ_API_KEY."
        )
    raise RuntimeError(
        "Could not transcribe the recording: " + "; ".join(errors)
        + ". Install `faster-whisper` to transcribe on this PC instead."
    )
