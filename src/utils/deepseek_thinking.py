"""
ChatDeepSeekThinking — a thin subclass of ChatDeepSeek that correctly
echoes `reasoning_content` back to the DeepSeek API on every turn.

Problem:
    DeepSeek's thinking-enabled models (e.g. deepseek-v4-pro with
    `extra_body={"thinking": {"type": "enabled"}}`) require that the
    `reasoning_content` field be included in the assistant message of every
    subsequent request.  Without it the API returns:

        400 The `reasoning_content` in the thinking mode must be passed
        back to the API.

    `langchain-deepseek` already stores the value in
    `AIMessage.additional_kwargs["reasoning_content"]`, but the base
    `_get_request_payload` serialiser does not re-inject it when it
    rebuilds the messages list for the next call.

Fix:
    Override `_get_request_payload` and, for every assistant message that
    carries a non-empty `reasoning_content` in its `additional_kwargs`,
    inject the field into the serialised payload dict before it is sent.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek


class ChatDeepSeekThinking(ChatDeepSeek):
    """ChatDeepSeek with multi-turn reasoning_content passthrough support.

    Usage::

        llm = ChatDeepSeekThinking(
            model="deepseek-v4-pro",
            temperature=0,
            extra_body={"thinking": {"type": "enabled"}},
        )
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Resolve the original BaseMessage list so we can look up additional_kwargs.
        # BaseChatOpenAI stores the converted messages in payload["messages"] as plain
        # dicts, but we need the originals to read additional_kwargs.
        from langchain_core.messages import BaseMessage
        from langchain_core.language_models import LanguageModelInput

        # Re-resolve the input to get the original BaseMessage objects.
        original_messages: list[BaseMessage] = self._convert_input(input_).to_messages()

        serialised = payload["messages"]

        # Walk both lists in lock-step; they should have identical length.
        for orig, serial in zip(original_messages, serialised):
            if serial.get("role") != "assistant":
                continue

            reasoning = orig.additional_kwargs.get("reasoning_content")
            if reasoning:
                serial["reasoning_content"] = reasoning

        return payload
