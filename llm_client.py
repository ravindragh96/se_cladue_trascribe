"""Databricks LLM response adapter. No revenue-classification business logic."""
import logging
import re
from typing import Any, Dict, List, Optional
from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """The model response is empty, malformed, or explicitly truncated."""


class LLMClient:
    def __init__(self, registry: ModelRegistry, endpoint_name: Optional[str] = None):
        self.registry = registry
        self.endpoint_name = endpoint_name

    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    @classmethod
    def _content_to_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                else:
                    kind = cls._get(item, "type")
                    if kind in (None, "text", "output_text"):
                        value = cls._get(item, "text")
                        if isinstance(value, str):
                            parts.append(value)
                        elif value is not None:
                            nested = cls._get(value, "value")
                            if isinstance(nested, str):
                                parts.append(nested)
            return "\n".join(parts)
        return ""

    @classmethod
    def _extract_text(cls, response: Any) -> str:
        if isinstance(response, str):
            return cls._clean_response(response)
        choices = cls._get(response, "choices")
        if choices:
            choice = choices[0]
            finish = cls._get(choice, "finish_reason")
            if finish in ("length", "max_tokens"):
                raise LLMResponseError("Model generation was truncated (finish_reason=length).")
            message = cls._get(choice, "message")
            content = cls._get(message, "content") if message is not None else None
            text = cls._content_to_text(content)
            if not text:
                text = cls._content_to_text(cls._get(choice, "text"))
            if text.strip():
                return cls._clean_response(text)
        predictions = cls._get(response, "predictions")
        if predictions:
            first = predictions[0]
            if isinstance(first, str):
                return cls._clean_response(first)
            for key in ("content", "text", "generated_text"):
                text = cls._content_to_text(cls._get(first, key))
                if text.strip():
                    return cls._clean_response(text)
        for key in ("content", "text", "generated_text", "output_text"):
            text = cls._content_to_text(cls._get(response, key))
            if text.strip():
                return cls._clean_response(text)
        raise LLMResponseError("No usable generated text in model response.")

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text,
                      flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text,
                             flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.0, max_tokens: int = 1500) -> str:
        messages = self._build_messages(system_prompt, user_prompt)
        response = self.registry.predict_llm(
            messages=messages, temperature=temperature,
            max_tokens=max_tokens, endpoint_name=self.endpoint_name,
        )
        text = self._extract_text(response)
        if not text:
            raise LLMResponseError("Model returned empty text after cleanup.")
        logger.info("LLM text extracted successfully; characters=%d", len(text))
        return text




##########################
"""
==============================================================================
Track A - Service Experts Revenue Opportunity Detection

LLM Client

Responsibilities
----------------
1. Build chat messages
2. Call ModelRegistry
3. Normalize Databricks Foundation Model responses
4. Return clean text

Business logic is NOT handled here.

Author
------
Ravindra Gantasala
==============================================================================
"""

import logging
import re
from typing import Dict, List, Optional

from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Centralized client for all LLM inference.

    Every downstream module should call ONLY this class.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        endpoint_name: Optional[str] = None
    ):

        self.registry = registry
        self.endpoint_name = endpoint_name

        logger.info("LLM Client initialized.")

    # ------------------------------------------------------------------
    # Build Messages
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> List[Dict]:

        return [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ]

    # ------------------------------------------------------------------
    # Public Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:

        messages = self._build_messages(
            system_prompt,
            user_prompt
        )

        raw_response = self.registry.predict_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint_name=self.endpoint_name
        )

        return self._extract_text(raw_response)

    # ------------------------------------------------------------------
    # Databricks Response Parser
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        """
        Normalizes every Databricks response shape.

        Supported

        Chat Endpoint

        {
            "choices":[
                {
                    "message":{
                        "content":"..."
                    }
                }
            ]
        }

        Foundation Models

        {
            "predictions":[
                {
                    "content":"..."
                }
            ]
        }

        {
            "predictions":[
                {
                    "text":"..."
                }
            ]
        }

        Legacy

        {
            "predictions":[ "..."]
        }
        """

        # The model output may repeat PII from the call. Never log it verbatim.
        logger.debug("LLM response received, type=%s", type(response).__name__)

        text = None

        try:

            # ----------------------------------------------------------
            # OpenAI Chat schema
            # ----------------------------------------------------------

            if isinstance(response, dict):

                choices = response.get("choices")

                if choices:

                    message = choices[0].get("message", {})

                    text = message.get("content")

                # ------------------------------------------------------
                # Databricks Foundation Model schema
                # ------------------------------------------------------

                if text is None:

                    predictions = response.get("predictions")

                    if predictions:

                        first = predictions[0]

                        if isinstance(first, dict):

                            text = (
                                first.get("content")
                                or first.get("text")
                                or first.get("generated_text")
                            )

                        elif isinstance(first, str):

                            text = first

                # ------------------------------------------------------
                # Direct Content
                # ------------------------------------------------------

                if text is None:

                    text = (
                        response.get("content")
                        or response.get("text")
                    )

            elif isinstance(response, str):

                text = response

            else:

                text = str(response)

        except Exception:

            logger.exception("Unable to parse model response.")

            text = str(response)

        if text is None:
            text = ""

        return self._clean_response(text)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _clean_response(
        self,
        text: str
    ) -> str:

        if not text:
            return ""

        if isinstance(text, list):
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )

        text = text.strip()

        text = re.sub(
            r"^```json",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^```",
            "",
            text
        )

        text = re.sub(
            r"```$",
            "",
            text
        )

        # Remove complete reasoning/thinking blocks, not only their markers.
        # A leftover reasoning narrative would otherwise make JSON parsing fail.
        text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)

        return text.strip()
