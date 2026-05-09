# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Refacer is a fully-offline desktop tool for batch face anonymization and metadata scrubbing, targeted at photojournalists and activist photographers. It detects faces in images, replaces each with a randomly generated AI identity (non-reversible, non-deterministic), and strips all EXIF/XMP/IPTC metadata via `exiftool`. No network calls happen at runtime.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python scripts/download_models.py   # fetches GFPGANv1.4.pth automatically
# Then manually place inswapper_128.onnx into models/ (Google Drive link in README)
```

`exiftool` must be installed separately and on PATH (Windows: download from exiftool.org; macOS: `brew install exiftool`; Linux: `apt install libimage-exiftool-perl`). Without it, metadata scrubbing is silently skipped and a prominent warning is logged.

## Running

**UI:**
```bash
python ui/app.py
# Opens Gradio at http://127.0.0.1:7860
```

**CLI:**
```bash
python -m refacer --input /path/to/photos --output /path/to/output
python -m refacer --input /path/to/photos --output /path/to/output --log-level DEBUG
python -m refacer --help
```

## Architecture

The project is a flat Python package at the repo root (not under a `src/` directory). The main modules are:

| File | Role |
|---|---|
| `pipeline.py` | Batch orchestrator — iterates images, calls swap/enhancement/scrub, collects `RunStats` |
| `models.py` | Loads all ML models once at startup into a `ModelBundle` dataclass |
| `swap.py` | Per-face logic: random latent generation, ONNX inference, colour correction, compositing |
| `metadata.py` | Wraps `exiftool` for scrubbing and post-scrub verification |
| `__main__.py` | CLI entrypoint (`argparse`) |
| `ui/app.py` | Gradio UI — loads models at startup, calls `pipeline.run()` on button click |
| `scripts/download_models.py` | One-shot script to download GFPGAN weights |

**Pipeline flow per image:**

```
cv2.imread → FaceAnalysis.get() → [per face: swap_face()] → GFPGANer.enhance() → cv2.imwrite(temp) → exiftool scrub → verify → os.replace(temp → final)
```

**Resilience contract** (defined in `pipeline.py` docstring):
- Per-face swap failure → partial save (other faces still swapped), logged
- Whole-image detection failure → original copied to output unchanged
- GFPGAN failure → saved without enhancement
- Metadata scrub/verify failure → output **discarded** (never publish unscrubbed images)

**Model files** (must exist in `models/` before running):
- `inswapper_128.onnx` — face swapper (manual download required)
- `GFPGANv1.4.pth` — face enhancer (fetched by `download_models.py`)

InsightFace downloads `buffalo_l` detection weights to its own cache on first run (~300 MB).

**Key design decisions:**
- All inference runs on CPU via `onnxruntime` with `CPUExecutionProvider` — no GPU required
- `ModelBundle` is loaded once at startup and passed through the call chain; never reloaded per image
- Random identity in `swap.py` uses `np.random.randn(512)` with no seed — each face gets a unique, unreproducible replacement
- Images are written to a `.tmp` file first, then scrubbed, verified, and atomically promoted via `os.replace()` — a corrupt or unscrubbed temp file is never exposed as output
- `ui/app.py` manually inserts the repo parent onto `sys.path` since it lives one level below the package root
