"""Gemma 4 response parser + chat-handler subclass with output post-processing.

llama-cpp-python's `Gemma4ChatHandler` (JamePeng fork) renders the *input*
prompt with Gemma 4's chat template, but the response path ends at
`_convert_completion_to_chat` without unpacking Gemma 4's special-token
sequences. The model's reasoning channel and tool calls therefore arrive as
raw text in `message.content`:

    <|channel>thought
    ...reasoning...
    <channel|>
    <|tool_call>call:NAME{key:value,...}<tool_call|>

This module fills that gap. It exposes:

  - `parse_assistant_message(raw)`: split raw text into OpenAI-shaped
    `content`, `reasoning_content`, and `tool_calls`.
  - `Gemma4ToolChatHandler`: subclass of `Gemma4ChatHandler` that runs the
    parser on non-streaming responses, so downstream consumers (e.g.
    `caption.py`'s tool loop) see a normal structured message.

Gemma 4 expects its prior thought to be replayed alongside the tool_call in
subsequent turns — the chat template's `<|channel>thought\\n... <channel|>`
emission is guarded by `message.get('tool_calls')`. As long as we populate
both `reasoning_content` and `tool_calls` on the assistant message and pass
them back in, the round-trip works.

Argument wire format (from Gemma4ChatHandler.CHAT_FORMAT, `escape_keys=False`
inside tool_call args and tool_response bodies):

    <|"|>STRING<|"|>      string
    [v, v, ...]           array
    {k:v, ...}            object (bare keys)
    true / false / null   booleans / null
    123 / 1.5             number
    BARE_IDENT            bare identifier (rare; some enum slots)
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from llama_cpp.llama_chat_format import Gemma4ChatHandler

_STR_DELIM = '<|"|>'
_THOUGHT_RE = re.compile(r"<\|channel>thought\n?(.*?)\n?<channel\|>", re.DOTALL)
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL
)
_TRAILING_STOP_RE = re.compile(r"(?:<\|tool_response>|<turn\|>|<eos>)\s*$")


class _ArgParser:
    """Recursive-descent parser for Gemma 4's tool-call argument syntax."""

    def __init__(self, src: str):
        self.src = src
        self.pos = 0

    def parse(self) -> Any:
        value = self._value()
        self._ws()
        if self.pos != len(self.src):
            tail = self.src[self.pos : self.pos + 20]
            raise ValueError(f"trailing input at pos {self.pos}: {tail!r}")
        return value

    def _ws(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] in " \t\r\n":
            self.pos += 1

    def _peek(self, n: int = 1) -> str:
        return self.src[self.pos : self.pos + n]

    def _try(self, s: str) -> bool:
        if self.src.startswith(s, self.pos):
            self.pos += len(s)
            return True
        return False

    def _value(self) -> Any:
        self._ws()
        if self._peek(len(_STR_DELIM)) == _STR_DELIM or self._peek() == '"':
            return self._string()
        head = self._peek()
        if head == "[":
            return self._array()
        if head == "{":
            return self._object()
        return self._bare()

    def _string(self) -> str:
        # Gemma-native form: <|"|>...<|"|>
        if self._try(_STR_DELIM):
            end = self.src.find(_STR_DELIM, self.pos)
            if end < 0:
                raise ValueError(f"unterminated <|\"|>-string at pos {self.pos}")
            value = self.src[self.pos : end]
            self.pos = end + len(_STR_DELIM)
            return value
        # JSON fallback: "..." with backslash escapes (the model sometimes
        # emits this when it gets confused — accept it rather than dying).
        if self._try('"'):
            escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
            out: list[str] = []
            while self.pos < len(self.src):
                ch = self.src[self.pos]
                if ch == '"':
                    self.pos += 1
                    return "".join(out)
                if ch == "\\" and self.pos + 1 < len(self.src):
                    out.append(escapes.get(self.src[self.pos + 1], self.src[self.pos + 1]))
                    self.pos += 2
                    continue
                out.append(ch)
                self.pos += 1
            raise ValueError(f"unterminated \"-string at pos {self.pos}")
        raise ValueError(f"expected string at pos {self.pos}")

    def _array(self) -> list:
        if not self._try("["):
            raise ValueError(f"expected [ at pos {self.pos}")
        out: list = []
        self._ws()
        if self._try("]"):
            return out
        while True:
            out.append(self._value())
            self._ws()
            if self._try(","):
                continue
            if self._try("]"):
                return out
            raise ValueError(f"expected , or ] at pos {self.pos}")

    def _object(self) -> dict:
        if not self._try("{"):
            raise ValueError(f"expected {{ at pos {self.pos}")
        out: dict = {}
        self._ws()
        if self._try("}"):
            return out
        while True:
            self._ws()
            if self._peek(len(_STR_DELIM)) == _STR_DELIM or self._peek() == '"':
                key = self._string()
            else:
                start = self.pos
                while (
                    self.pos < len(self.src)
                    and self.src[self.pos] not in ":, \t\r\n"
                ):
                    self.pos += 1
                key = self.src[start : self.pos]
                if not key:
                    raise ValueError(f"empty key at pos {self.pos}")
            self._ws()
            if not self._try(":"):
                raise ValueError(f"expected : after key {key!r} at pos {self.pos}")
            out[key] = self._value()
            self._ws()
            if self._try(","):
                continue
            if self._try("}"):
                return out
            raise ValueError(f"expected , or }} at pos {self.pos}")

    def _bare(self) -> Any:
        start = self.pos
        # Bare values run until the next structural delimiter. Newlines DON'T
        # terminate — Gemma's renderer never inserts them mid-arg, but stray
        # whitespace gets trimmed below.
        while self.pos < len(self.src) and self.src[self.pos] not in ",]}":
            self.pos += 1
        tok = self.src[start : self.pos].strip()
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok == "null":
            return None
        try:
            return int(tok)
        except ValueError:
            try:
                return float(tok)
            except ValueError:
                return tok  # bare identifier (e.g. uppercase enum)


def _parse_tool_args(body: str) -> dict:
    """Parse the {...} body of a tool call into a Python dict."""
    parsed = _ArgParser("{" + body + "}").parse()
    if not isinstance(parsed, dict):
        return {"_value": parsed}
    return parsed


def parse_assistant_message(raw: str) -> dict:
    """Convert Gemma 4 raw assistant output into an OpenAI-style message dict.

    Returns a dict with `role="assistant"` plus, where applicable,
    `content` (None when nothing remained after stripping), `reasoning_content`
    (joined with blank lines if multiple thought blocks), and `tool_calls`
    (OpenAI shape, with synthesized `call_*` IDs since Gemma's wire format
    doesn't carry one).
    """
    raw = _TRAILING_STOP_RE.sub("", raw)
    thoughts = _THOUGHT_RE.findall(raw)
    tool_calls_raw = _TOOL_CALL_RE.findall(raw)
    leftover = _TOOL_CALL_RE.sub("", _THOUGHT_RE.sub("", raw)).strip()

    msg: dict = {"role": "assistant"}
    if thoughts:
        msg["reasoning_content"] = "\n\n".join(t.strip() for t in thoughts)
    if tool_calls_raw:
        tool_calls = []
        for name, body in tool_calls_raw:
            parse_error: str | None = None
            try:
                args = _parse_tool_args(body)
            except Exception as exc:  # noqa: BLE001
                # Don't leak the malformed args back into the replay — that
                # creates a feedback loop where the model sees its own garbage
                # rerendered and tries even more variants. Replace with `{}`
                # and surface the error on the tool_call itself; caption.py
                # turns this into a format-hint tool response.
                args = {}
                parse_error = str(exc)
            tc = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
            if parse_error is not None:
                tc["_parse_error"] = parse_error
            tool_calls.append(tc)
        msg["tool_calls"] = tool_calls
    msg["content"] = leftover if leftover else None
    return msg


class Gemma4ToolChatHandler(Gemma4ChatHandler):
    """`Gemma4ChatHandler` + automatic post-parse of Gemma 4 special tokens.

    Non-streaming responses get their `message.content` split into structured
    `reasoning_content`, `tool_calls`, and a trimmed `content`. Streaming
    responses pass through unchanged.
    """

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        if kwargs.get("stream"):
            return result
        try:
            choice = result["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError):
            return result
        raw = msg.get("content")
        if not isinstance(raw, str):
            return result
        parsed = parse_assistant_message(raw)
        msg["content"] = parsed["content"]
        if "reasoning_content" in parsed:
            msg["reasoning_content"] = parsed["reasoning_content"]
        if "tool_calls" in parsed:
            msg["tool_calls"] = parsed["tool_calls"]
            if choice.get("finish_reason") in (None, "stop"):
                choice["finish_reason"] = "tool_calls"
        return result
