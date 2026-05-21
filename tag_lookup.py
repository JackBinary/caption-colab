"""In-process semantic tag lookup over the prebuilt danbooru-db index.

Trimmed copy of the query path from the danbooru-db repo. Index creation
lives in that repo; this file only knows how to *read*: load the Q8_0
Harrier embedding model, embed a query, and ANN-search the sqlite-vec
table that ships inside danbooru.db.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec
from huggingface_hub import hf_hub_download
from llama_cpp.llama_embedding import LlamaEmbedding

EMBED_DIM = 640

QUERY_MODEL_REPO = "mykor/harrier-oss-v1-270m-GGUF"
QUERY_MODEL_FILE = "harrier-oss-v1-270M-Q8_0.gguf"

# Harrier expects this instruction prefix on queries (not on docs).
QUERY_INSTRUCTION = (
    "Given a natural-language description of an image, retrieve Danbooru "
    "tags whose wiki pages describe matching visual content."
)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _load_query_model(n_gpu_layers: int = -1) -> LlamaEmbedding:
    # JamePeng's fork deprecated `Llama.create_embedding` and the deprecated
    # path is broken (calls `LlamaBatch.add_sequence` without the new
    # `logits_array` arg). `LlamaEmbedding` is the supported replacement and
    # auto-sets embeddings=True.
    path = hf_hub_download(QUERY_MODEL_REPO, QUERY_MODEL_FILE)
    return LlamaEmbedding(
        model_path=path,
        n_ctx=4096,
        pooling_type=3,  # LAST
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def _snippet(body: str | None, width: int = 200) -> str:
    if not body:
        return ""
    s = " ".join(body.split())
    return s[:width] + ("…" if len(s) > width else "")


class TagLookup:
    """Thread-safe in-process tag lookup.

    Captioning workers share one instance; the lock serialises both the
    embed call (single Llama context) and the sqlite connection."""

    def __init__(self, db_path: str | Path, n_gpu_layers: int = -1):
        self.conn = _connect(db_path)
        self.model = _load_query_model(n_gpu_layers=n_gpu_layers)
        self.lock = threading.Lock()

    def lookup(self, concept: str, k: int = 5) -> list[dict[str, Any]]:
        prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {concept}"
        with self.lock:
            out = self.model.create_embedding(prompt)
            vec = np.asarray(out["data"][0]["embedding"], dtype=np.float32)
            if vec.ndim == 2:
                vec = vec[-1]
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            qblob = struct.pack(f"{EMBED_DIM}f", *vec.tolist())
            rows = self.conn.execute(
                """
                SELECT t.name, t.post_count, v.distance, t.body_clean
                FROM vec_tags v
                JOIN tags t ON t.rowid = v.rowid
                WHERE v.embedding MATCH ?
                  AND k = ?
                ORDER BY v.distance
                """,
                (qblob, k),
            ).fetchall()
        return [
            {
                "name": name,
                "post_count": post_count,
                "distance": round(float(dist), 4),
                "snippet": _snippet(body),
            }
            for (name, post_count, dist, body) in rows
        ]
