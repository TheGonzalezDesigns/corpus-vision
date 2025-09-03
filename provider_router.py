import os
import logging
from typing import Optional, List

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

try:
    import anthropic
except Exception:
    anthropic = None

from PIL import Image


class VisionProvider:
    name: str = "base"

    def available(self) -> bool:
        return False

    def analyze(self, image: Image.Image, prompt: str) -> Optional[str]:
        raise NotImplementedError


class GeminiVision(VisionProvider):
    name = "gemini"

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_id = os.environ.get("GEMINI_VISION_MODEL", "gemini-1.5-flash")
        self._model = None
        if self.available():
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_id)
            except Exception as e:
                logging.error(f"Gemini init failed: {e}")
                self._model = None

    def available(self) -> bool:
        return bool(self.api_key and genai is not None)

    def analyze(self, image: Image.Image, prompt: str) -> Optional[str]:
        if not self._model:
            return None
        try:
            resp = self._model.generate_content([prompt, image])
            return (resp.text or "").strip() if resp else None
        except Exception as e:
            logging.warning(f"Gemini error: {e}")
            return None


class OpenAIVision(VisionProvider):
    name = "openai"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_id = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
        self._client = None
        if self.available():
            try:
                self._client = OpenAIClient(api_key=self.api_key)
            except Exception as e:
                logging.error(f"OpenAI init failed: {e}")
                self._client = None

    def available(self) -> bool:
        return bool(self.api_key and OpenAIClient is not None)

    def analyze(self, image: Image.Image, prompt: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            import base64, io
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            resp = self._client.responses.create(
                model=self.model_id,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_data": b64,
                                "mime_type": "image/jpeg",
                            },
                        ],
                    }
                ],
            )
            if hasattr(resp, "output_text"):
                return (resp.output_text or "").strip()
            return None
        except Exception as e:
            logging.warning(f"OpenAI Vision error: {e}")
            return None


class ClaudeVision(VisionProvider):
    name = "claude"

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.model_id = os.environ.get("CLAUDE_VISION_MODEL", "claude-3-5-sonnet-latest")
        self._client = None
        if self.available():
            try:
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logging.error(f"Anthropic init failed: {e}")
                self._client = None

    def available(self) -> bool:
        return bool(self.api_key and anthropic is not None)

    def analyze(self, image: Image.Image, prompt: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            import base64, io
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            msg = self._client.messages.create(
                model=self.model_id,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                            },
                        ],
                    }
                ],
            )
            if msg and msg.content:
                texts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
                return "\n".join(texts).strip() if texts else None
            return None
        except Exception as e:
            logging.warning(f"Claude Vision error: {e}")
            return None


class VisionRouter:
    def __init__(self, priority: Optional[List[str]] = None):
        order = priority or os.environ.get("VISION_PROVIDER_ORDER", "gemini,openai,claude").split(",")
        order = [x.strip().lower() for x in order if x.strip()]
        self.providers: List[VisionProvider] = []
        for name in order:
            if name == "gemini":
                self.providers.append(GeminiVision())
            elif name == "openai":
                self.providers.append(OpenAIVision())
            elif name == "claude":
                self.providers.append(ClaudeVision())

    def analyze(self, image: Image.Image, prompt: str) -> Optional[str]:
        for provider in self.providers:
            if not provider.available():
                continue
            result = provider.analyze(image, prompt)
            if result:
                logging.info(f"VisionRouter: provider '{provider.name}' succeeded")
                return result
            else:
                logging.info(f"VisionRouter: provider '{provider.name}' failed, trying next")
        return None
