"""OpenRouter chat-completions client using Python's standard library."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multilingual_rag.providers.base import ChatMessage, ProviderError


logger = logging.getLogger(__name__)


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        endpoint: str,
        timeout_seconds: float = 45,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenRouter API key cannot be empty")
        if not model.strip():
            raise ValueError("OpenRouter model cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def generate(self, messages: list[ChatMessage]) -> str:
        payload = json.dumps({"model": self._model, "messages": messages}).encode("utf-8")
        request = Request(
            self._endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            logger.warning(
                "OpenRouter request failed",
                extra={"status_code": exc.code, "model": self._model},
            )
            raise ProviderError(
                f"OpenRouter returned HTTP {exc.code}. Check model access and account limits."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            logger.warning("OpenRouter connection failed", extra={"model": self._model})
            raise ProviderError("Could not reach OpenRouter; check connectivity and try again.") from exc
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            logger.exception("OpenRouter returned an invalid response", extra={"model": self._model})
            raise ProviderError("OpenRouter returned an invalid chat-completion response.") from exc

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.exception("OpenRouter response did not contain a message", extra={"model": self._model})
            raise ProviderError("OpenRouter returned no assistant message.") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("OpenRouter returned an empty assistant message.")
        return content.strip()