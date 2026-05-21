# caption-colab

Anima-style image captioning in a Colab notebook. A local llama.cpp VLM (Gemma 4 26B-A4B) iteratively grounds each visual concept against a semantic Danbooru tag index, then emits a natural-language prompt. Each image in `./source/` gets a sidecar `.txt`.

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackBinary/caption-colab/blob/main/caption_colab.ipynb)

## Runtime requirements

The VLM is 26B params at Q8 — about 28 GB on disk and ~30 GB in VRAM. You need **Colab Pro+ with an A100 (40 GB)** or equivalent. The Harrier embedding model (~270 MB) shares the same GPU.

## What's in this repo

| File | Purpose |
|---|---|
| `caption_colab.ipynb` | The notebook. Open in Colab, set GPU, run cells. |
| `caption.py` | Captioning pipeline — VLM tool-calling loop, ThreadPool, tqdm UI. |
| `tag_lookup.py` | In-process semantic tag lookup over the sqlite-vec index. |
| `Anima_prompting.md` | System prompt that defines Anima caption style. |
| `requirements.txt` | Python deps (excluding `llama-cpp-python`, installed separately). |
| `source/` | Drop images here. The notebook writes `<name>.txt` next to each. |

## Architecture

```
            ┌──── caption_colab.ipynb (in Colab) ─────┐
            │                                          │
 source/ ─► │ caption.py                               │
            │   ├─ ThreadPool(N) ─► llama-server ─────┼─► Gemma 4 26B VLM
            │   │                   (localhost:8080)   │   (tool-calling)
            │   │                                      │
            │   └─ tag_lookup.py ─► Harrier embed ─────┼─► sqlite-vec
            │                       danbooru.db        │   (ANN search)
            └──────────────────────────────────────────┘
```

For every image, the VLM enumerates visual concepts, calls `lookup_tags` once to ground them against canonical Danbooru wording, then calls `submit_caption` with the final Anima prompt. Both tools are local — no external API calls beyond initial artifact downloads.
