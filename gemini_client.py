"""
gemini_client.py
Cloud LLM interface using Google Gemini API for NEXUS AI.
"""

import os
from pathlib import Path

import requests
from typing import Optional

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Entered once in the UI, remembered here so the cloud path is available on
# every later run without the user re-pasting it.
KEY_FILE = Path(__file__).resolve().parent / ".gemini_key"


def load_saved_key() -> Optional[str]:
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def save_key(key: str) -> None:
    key = (key or "").strip()
    if key:
        KEY_FILE.write_text(key, encoding="utf-8")
    elif KEY_FILE.exists():
        KEY_FILE.unlink()


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or load_saved_key()

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(self, prompt: str, model: str = DEFAULT_GEMINI_MODEL, system_instruction: Optional[str] = None) -> str:
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not set. Please set the API key in environment or UI sidebar.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"System Instruction: {system_instruction}"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {"contents": contents}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API request failed ({response.status_code}): {response.text}")

        res_json = response.json()
        try:
            candidates = res_json.get("candidates", [])
            if not candidates:
                return "Gemini returned no candidates."
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts).strip()
        except KeyError as exc:
            raise RuntimeError(f"Unexpected response structure from Gemini API: {res_json}") from exc
