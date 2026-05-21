"""Caption every image in ./source with an Anima-style prompt.

Pipeline per image:
  1. The image is handed to a local llama-cpp-python VLM (in-process,
     loaded with a `Gemma4ChatHandler`) along with the rules in
     Anima_prompting.md as the system prompt.
  2. The VLM has a `lookup_tags` tool that performs semantic search over
     the danbooru-db vector index (general-category tags only — no
     copyrights or character names).
  3. The model iterates — calling the tool to ground each visual concept
     against canonical Danbooru wording — and emits the final prompt via
     `submit_caption`, which is written as a sidecar .txt next to the image.

Captioning is serial: llama-cpp-python holds a single Llama instance and
its inference call is not safely concurrent, so images are processed one
at a time.
"""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from tag_lookup import TagLookup

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_TOOL_ITERATIONS = 32


# --------------------------------------------------------------------- VLM I/O

def _image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_tags",
            "description": (
                "Batch semantic search over the Danbooru general-tag wiki "
                "index. Pass a LIST of visual concepts (e.g. ['red "
                "oil-paper umbrella', 'long blonde hair with bangs', "
                "'silver pauldrons']) and get back the closest canonical "
                "tags for each, ranked by relevance. The index only "
                "contains general-category tags — character names, artist "
                "names, and copyrights are not present.\n\n"
                "Prefer one batched call over many sequential calls — "
                "list every concept you want grounded in a single "
                "invocation. You can still call again later if you spot "
                "concepts you missed.\n\n"
                "If a concept returns no close match (high distance, "
                "irrelevant names), that's fine: Anima still accepts "
                "plain-English description for that part of the caption. "
                "Tags are preferred but not required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Visual concepts to search for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Top matches per concept (default 5).",
                    },
                },
                "required": ["concepts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_caption",
            "description": (
                "Submit the final Anima-style caption for the image. Call "
                "this exactly once, after every visual concept has been "
                "grounded with lookup_tags. Calling this ends the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "caption": {
                        "type": "string",
                        "description": (
                            "The final Anima prompt — just the prompt text, "
                            "no preamble, no quotes, no markdown."
                        ),
                    },
                },
                "required": ["caption"],
            },
        },
    },
]


TOOL_PROTOCOL_PROMPT = """You are a vision captioner. For every image you are shown, produce exactly one Anima-style prompt and submit it via the `submit_caption` tool.

## Required tool-use protocol

1. **Enumerate the image in natural language.** For each thing you intend to put in the caption — body type, hair (length, color, style, accessories), eyes, expression, each clothing piece with color and material, pose, action, props, background — write a short descriptive phrase the way a person would actually describe it out loud: `"long brown hair tied in a high ponytail with bangs"`, `"black hoodie with orange and purple stripes"`, `"abstract cyberpunk neon background with cyan and magenta lights"`. Do NOT pre-guess Danbooru tags here — describe what you see; the lookup step does the tag mapping.

2. **Ground the concepts.** Call `lookup_tags` ONCE, passing that list of natural-language phrases as `concepts`. `lookup_tags` performs semantic search against the Danbooru tag wiki and returns the closest canonical tag names for each phrase. The index is general-category only — character names, artist names, and copyrights are not present, so it cannot leak them.

3. **Write the caption** using the grounded wording. Where `lookup_tags` returned no close match (high distance, irrelevant names), describe that part in plain English; Anima accepts natural language for those slots.

4. **Submit.** Call `submit_caption` exactly once with the final prompt as the `caption` argument.

## Rules

- `lookup_tags` is a **natural-language semantic search**, not a tag-to-tag lookup. Send rich descriptive phrases, not bare tags. `"red oil-paper umbrella"` is a good query; `"umbrella"` is too generic to ground well.
- You MUST call `lookup_tags` at least once before calling `submit_caption`. `submit_caption` will be rejected until at least one batched lookup has happened.
- Prefer one batched `lookup_tags` call over many sequential calls. List every concept in a single invocation; you can call again later if you spot something you missed.
- Do not skip the protocol even if you already "know" the Danbooru wording — ground it.
- Do not emit the caption as plain text. Submit it only through `submit_caption`.

# Anima Style Guide

"""


def _strip_frontmatter(text: str) -> str:
    """Strip a leading YAML frontmatter block (`--- ... ---`) if present."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        return text
    return text[end + len("\n---\n") :].lstrip()


def build_system_prompt(anima_md: str) -> str:
    return TOOL_PROTOCOL_PROMPT + _strip_frontmatter(anima_md)


def build_initial_messages(system_prompt: str, image_path: Path) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Caption this image."},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
            ],
        },
    ]


# ----------------------------------------------------------------- captioning

def _chat(messages: list[dict], llm: Any) -> dict:
    """Non-streaming chat completion. Returns the assistant message dict."""
    resp = llm.create_chat_completion(
        messages=messages,
        tools=TOOL_DEFS,
        tool_choice="auto",
        temperature=0.4,
        stream=False,
    )
    return resp["choices"][0]["message"]


def _normalize_tool_call(tc: Any) -> tuple[str, str, dict]:
    call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
    fn = tc.get("function", {})
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
    else:
        args = raw_args or {}
    return call_id, name, args


def _replay_assistant(msg: dict) -> dict:
    # The Gemma chat template only re-renders reasoning_content when the
    # assistant turn carries tool_calls; drop it on terminal turns.
    if msg.get("tool_calls"):
        return msg
    return {k: v for k, v in msg.items() if k != "reasoning_content"}


def caption_image(image_path: Path, system_prompt: str, lookup: TagLookup,
                  llm: Any) -> tuple[str, int]:
    messages = build_initial_messages(system_prompt, image_path)
    lookups = 0

    for _ in range(MAX_TOOL_ITERATIONS):
        msg = _chat(messages, llm)
        messages.append(_replay_assistant(msg))

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            text = (msg.get("content") or "").strip()
            if text:
                return text, lookups
            raise RuntimeError("Assistant returned neither tool_calls nor text.")

        submitted: str | None = None
        for tc in tool_calls:
            call_id, name, args = _normalize_tool_call(tc)
            if name == "submit_caption":
                candidate = (args.get("caption") or "").strip()
                if lookups == 0:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({
                            "ok": False,
                            "error": (
                                "submit_caption rejected: you must call "
                                "lookup_tags at least once before submitting. "
                                "Enumerate every visual concept in the image "
                                "and call lookup_tags with the full list, "
                                "then call submit_caption."
                            ),
                        }),
                    })
                    continue
                if not candidate:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({
                            "ok": False,
                            "error": (
                                "submit_caption rejected: caption was empty. "
                                "Write the Anima prompt and resubmit."
                            ),
                        }),
                    })
                    continue
                submitted = candidate
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"ok": True}),
                })
                continue
            if name == "lookup_tags":
                concepts = args.get("concepts") or []
                if isinstance(concepts, str):
                    concepts = [concepts]
                k = int(args.get("k", 5))
                batch: list[dict] = []
                for concept in concepts:
                    try:
                        results = lookup.lookup(concept, k)
                        batch.append({"concept": concept, "matches": results})
                        lookups += 1
                    except Exception as e:  # noqa: BLE001
                        batch.append({"concept": concept, "error": str(e)})
                content = json.dumps(batch)
            else:
                content = json.dumps({"error": f"unknown tool: {name}"})
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })

        if submitted is not None:
            return submitted, lookups

    raise RuntimeError("Exceeded max tool iterations without a final answer.")


# ----------------------------------------------------------------------- main

def caption_all(
    source_dir: str | Path,
    *,
    llm: Any,
    db_path: str | Path,
    anima_md_path: str | Path,
    overwrite: bool = False,
    n_gpu_layers: int = -1,
) -> tuple[int, int]:
    """Caption every image in `source_dir` whose sidecar .txt is missing.

    Returns (captioned_count, error_count).
    """
    source = Path(source_dir)
    images = sorted(p for p in source.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {source}")
        return 0, 0

    todo = [p for p in images
            if overwrite or not p.with_suffix(".txt").exists()]
    skipped = len(images) - len(todo)
    if skipped:
        print(f"Skipping {skipped} already-captioned image(s).")
    if not todo:
        return 0, 0

    system_prompt = build_system_prompt(Path(anima_md_path).read_text())
    print(f"Loading tag-lookup model + opening {db_path} ...", flush=True)
    lookup = TagLookup(db_path, n_gpu_layers=n_gpu_layers)

    done = 0
    errors = 0
    bar = tqdm(total=len(todo), unit="img", desc="caption")
    for path in todo:
        try:
            prompt, lookups = caption_image(path, system_prompt, lookup, llm)
            if not prompt:
                raise RuntimeError("Empty response from VLM.")
            path.with_suffix(".txt").write_text(prompt + "\n")
            done += 1
            tqdm.write(f"[OK]  {path.name}: {lookups} lookups → {len(prompt)} chars")
        except Exception as e:  # noqa: BLE001
            errors += 1
            tqdm.write(f"[ERR] {path.name}: {e}")
        bar.update(1)
    bar.close()
    print(f"Captioned {done}/{len(todo)} images ({errors} errors).")
    return done, errors
