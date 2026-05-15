# UI Refresh & Live Streaming Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `pipeline.run()` into a generator and rewrite the Gradio UI as a three-panel layout (Upload → Progress → Output) that streams live log updates and gallery thumbnails as each image finishes.

**Architecture:** `pipeline.run()` yields one `ImageResult` per image; callers (CLI and UI) iterate the generator and accumulate results. A new `RunStats.from_results()` classmethod rebuilds the aggregate. The UI `process()` function is a Gradio generator that pushes log text, gallery images, and three stat numbers to the browser after each yield.

**Tech Stack:** Python 3.10+, Gradio 4.x, pytest, `unittest.mock`

---

## File Map

| File | Action | What changes |
|---|---|---|
| `pipeline.py` | Modify | `run()` → generator; add `count_images()`; add `RunStats.from_results()` |
| `__main__.py` | Modify | Adapt call site to iterate generator; build `RunStats` at end |
| `ui/app.py` | Modify | Full layout rewrite; `process()` → generator; add helpers |
| `tests/__init__.py` | Create | Empty — marks `tests/` as a package |
| `tests/test_pipeline.py` | Create | All pipeline unit tests |

---

## Task 1: Create test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Verify pytest is available**

```bash
python -m pytest --version
```

Expected: `pytest 7.x.x` or higher. If missing:

```bash
pip install pytest
```

- [ ] **Step 2: Create the tests package**

Create `tests/__init__.py` as an empty file.

- [ ] **Step 3: Create `tests/test_pipeline.py` with imports**

```python
import os
from unittest.mock import MagicMock, patch

import pytest

from refacer.pipeline import (
    FaceResult,
    ImageResult,
    RunStats,
    count_images,
    run,
)
```

- [ ] **Step 4: Verify the import works**

```bash
python -m pytest tests/test_pipeline.py --collect-only
```

Expected: `no tests ran` (no test functions yet) with no import errors.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_pipeline.py
git commit -m "test: scaffold pipeline test module"
```

---

## Task 2: Add `RunStats.from_results()` classmethod

**Files:**
- Modify: `pipeline.py` (the `RunStats` dataclass, lines 106–130)
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestRunStatsFromResults:
    def _make_image_result(self, filename, success, faces_detected=0, face_successes=(), enhancement_ok=True, metadata_scrubbed=True):
        face_results = [FaceResult(index=i, success=ok) for i, ok in enumerate(face_successes)]
        return ImageResult(
            filename=filename,
            success=success,
            faces_detected=faces_detected,
            face_results=face_results,
            enhancement_ok=enhancement_ok,
            metadata_scrubbed=metadata_scrubbed,
        )

    def test_empty_results(self):
        stats = RunStats.from_results(total=0, results=[])
        assert stats.total == 0
        assert stats.saved == 0
        assert stats.skipped == 0
        assert stats.failed == 0
        assert stats.total_faces == 0
        assert stats.faces_swapped == 0
        assert stats.faces_failed == 0
        assert stats.image_results == []

    def test_saved_image_counted(self):
        r = self._make_image_result("a.jpg", success=True, faces_detected=2, face_successes=(True, True))
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.saved == 1
        assert stats.skipped == 0
        assert stats.failed == 0
        assert stats.total_faces == 2
        assert stats.faces_swapped == 2
        assert stats.faces_failed == 0

    def test_skipped_image_counted(self):
        r = self._make_image_result("b.jpg", success=True, faces_detected=0)
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.skipped == 1
        assert stats.saved == 0
        assert stats.failed == 0

    def test_failed_image_counted(self):
        r = self._make_image_result("c.jpg", success=False)
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.failed == 1
        assert stats.saved == 0
        assert stats.skipped == 0

    def test_partial_face_swap(self):
        r = self._make_image_result("d.jpg", success=True, faces_detected=3, face_successes=(True, False, True))
        stats = RunStats.from_results(total=1, results=[r])
        assert stats.faces_swapped == 2
        assert stats.faces_failed == 1
        assert stats.saved == 1

    def test_image_results_preserved(self):
        r1 = self._make_image_result("a.jpg", success=True, faces_detected=1, face_successes=(True,))
        r2 = self._make_image_result("b.jpg", success=False)
        stats = RunStats.from_results(total=2, results=[r1, r2])
        assert stats.image_results == [r1, r2]
        assert stats.total == 2

    def test_total_overrides_len_results(self):
        # total is passed explicitly so callers can set it from count_images()
        stats = RunStats.from_results(total=5, results=[])
        assert stats.total == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pipeline.py::TestRunStatsFromResults -v
```

Expected: `AttributeError: type object 'RunStats' has no attribute 'from_results'`

- [ ] **Step 3: Implement `RunStats.from_results()`**

In `pipeline.py`, add this classmethod inside the `RunStats` dataclass, after the `__str__` method (after line 130):

```python
    @classmethod
    def from_results(cls, total: int, results: "List[ImageResult]") -> "RunStats":
        stats = cls(total=total, image_results=list(results))
        for r in results:
            stats.total_faces += r.faces_detected
            stats.faces_swapped += r.faces_swapped
            stats.faces_failed += r.faces_failed
            if not r.success:
                stats.failed += 1
            elif r.faces_detected == 0:
                stats.skipped += 1
            else:
                stats.saved += 1
        return stats
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pipeline.py::TestRunStatsFromResults -v
```

Expected: all 7 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: add RunStats.from_results() classmethod"
```

---

## Task 3: Add `count_images()` public helper

**Files:**
- Modify: `pipeline.py` (after `_collect_images`, around line 142)
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestCountImages:
    def test_counts_supported_extensions(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.webp").touch()
        (tmp_path / "d.txt").touch()   # excluded
        assert count_images(str(tmp_path)) == 3

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "a.JPG").touch()
        (tmp_path / "b.JPEG").touch()
        (tmp_path / "c.PNG").touch()
        assert count_images(str(tmp_path)) == 3

    def test_empty_directory(self, tmp_path):
        assert count_images(str(tmp_path)) == 0

    def test_no_supported_files(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.csv").touch()
        assert count_images(str(tmp_path)) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pipeline.py::TestCountImages -v
```

Expected: `ImportError` — `count_images` not found in `refacer.pipeline`.

- [ ] **Step 3: Implement `count_images()`**

In `pipeline.py`, add this function immediately after `_collect_images` (after line 142):

```python
def count_images(input_dir: str) -> int:
    """Return the number of supported images in *input_dir*."""
    return len(_collect_images(input_dir))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pipeline.py::TestCountImages -v
```

Expected: all 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: add count_images() public helper"
```

---

## Task 4: Refactor `run()` to a generator

**Files:**
- Modify: `pipeline.py` (`run()` function, lines 320–368)
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestRunGenerator:
    def _fake_result(self, filename, success=True, faces_detected=1):
        return ImageResult(
            filename=filename,
            success=success,
            faces_detected=faces_detected,
            face_results=[FaceResult(0, True)] if faces_detected else [],
        )

    def test_run_yields_image_result_per_image(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "a.jpg").touch()
        (input_dir / "b.jpg").touch()

        r1 = self._fake_result("a.jpg")
        r2 = self._fake_result("b.jpg")

        with patch("refacer.pipeline._process_image", side_effect=[r1, r2]):
            results = list(run(str(input_dir), str(tmp_path / "out"), MagicMock()))

        assert results == [r1, r2]

    def test_run_yields_in_sorted_order(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (input_dir / name).touch()

        fake = [self._fake_result(n) for n in ["a.jpg", "b.jpg", "c.jpg"]]

        with patch("refacer.pipeline._process_image", side_effect=fake):
            results = list(run(str(input_dir), str(tmp_path / "out"), MagicMock()))

        assert [r.filename for r in results] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_run_empty_directory_yields_nothing(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()

        results = list(run(str(input_dir), str(tmp_path / "out"), MagicMock()))

        assert results == []

    def test_run_creates_output_dir(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"  # does not exist yet

        list(run(str(input_dir), str(output_dir), MagicMock()))

        assert output_dir.exists()

    def test_run_is_a_generator(self, tmp_path):
        import types
        input_dir = tmp_path / "in"
        input_dir.mkdir()

        gen = run(str(input_dir), str(tmp_path / "out"), MagicMock())

        assert isinstance(gen, types.GeneratorType)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pipeline.py::TestRunGenerator -v
```

Expected: `test_run_is_a_generator` fails (`AssertionError` — `run()` currently returns `RunStats`, not a generator). The other tests may partially pass or fail with type errors.

- [ ] **Step 3: Rewrite `run()` as a generator**

Replace the entire `run()` function in `pipeline.py` (lines 320–368) with:

```python
def run(
    input_dir: str,
    output_dir: str,
    models: ModelBundle,
) -> "Generator[ImageResult, None, None]":
    """
    Process all supported images in *input_dir*, writing results to *output_dir*.

    Yields one ImageResult per image as it finishes. Callers accumulate the
    yielded results and pass them to RunStats.from_results() to get aggregates.

    Parameters
    ----------
    input_dir  : str  — path to folder containing source images
    output_dir : str  — path to folder for anonymised output images
    models     : ModelBundle — pre-loaded models from refacer.models.load_models()
    """
    os.makedirs(output_dir, exist_ok=True)

    filenames = _collect_images(input_dir)
    if not filenames:
        logger.warning("No supported images found in %s", input_dir)
        return

    logger.info("Found %d image(s) to process in %s", len(filenames), input_dir)

    for filename in filenames:
        logger.info("── Processing: %s", filename)
        image_result = _process_image(filename, input_dir, output_dir, models)
        gc.collect()
        logger.info(image_result.summary())
        yield image_result
```

Also add `Generator` to the typing import at the top of `pipeline.py`. The current import is:

```python
from typing import List
```

Change it to:

```python
from typing import Generator, List
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pipeline.py::TestRunGenerator -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v
```

Expected: all tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: refactor pipeline.run() to a generator that yields ImageResult"
```

---

## Task 5: Adapt `__main__.py` to the generator

**Files:**
- Modify: `__main__.py` (the `main()` function, lines 86–99)

- [ ] **Step 1: Replace the pipeline call site in `main()`**

In `__main__.py`, replace lines 86–99:

```python
    # --- Run pipeline ---
    from refacer import pipeline

    stats = pipeline.run(
        input_dir=args.input,
        output_dir=args.output,
        models=models,
    )

    print(stats)

    # Exit non-zero if every image failed
    if stats.total > 0 and stats.failed == stats.total:
        sys.exit(1)
```

with:

```python
    # --- Run pipeline ---
    from refacer import pipeline
    from refacer.pipeline import RunStats

    results = []
    for img_result in pipeline.run(
        input_dir=args.input,
        output_dir=args.output,
        models=models,
    ):
        results.append(img_result)

    stats = RunStats.from_results(total=len(results), results=results)

    print(stats)

    # Exit non-zero if every image failed
    if stats.total > 0 and stats.failed == stats.total:
        sys.exit(1)
```

- [ ] **Step 2: Verify the CLI still imports cleanly**

```bash
python -m refacer --help
```

Expected: prints the argparse help message with no errors.

- [ ] **Step 3: Commit**

```bash
git add __main__.py
git commit -m "fix: adapt CLI to iterate pipeline generator"
```

---

## Task 6: Rewrite `ui/app.py` — helpers and layout

**Files:**
- Modify: `ui/app.py` (full rewrite of the UI layout section, lines 60–165)

This task replaces the entire UI below the model-loading block. Work top-to-bottom through the file.

- [ ] **Step 1: Update imports at the top of `ui/app.py`**

The current imports are:

```python
import logging
import os
import shutil
import sys
```

Add `subprocess` (needed for open-folder button):

```python
import logging
import os
import shutil
import subprocess
import sys
```

- [ ] **Step 2: Update the pipeline import**

The current import:

```python
from refacer.pipeline import SUPPORTED_EXTENSIONS, RunStats, run
```

Change to:

```python
from refacer.pipeline import SUPPORTED_EXTENSIONS, RunStats, count_images, run
```

- [ ] **Step 3: Replace the helper functions (lines 64–78)**

Delete `_format_stats` and `_list_output_images`. Replace with these four helpers:

```python
def _build_log(results: list) -> str:
    return "\n".join(r.summary() for r in results)


def _warning_count(results: list) -> int:
    return sum(
        1 for r in results
        if r.faces_failed > 0
        or not r.enhancement_ok
        or (metadata.is_available() and r.success and not r.metadata_scrubbed)
    )


def _list_output_images() -> list[str]:
    return sorted(
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def _clear_and_copy(input_files) -> None:
    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith(SUPPORTED_EXTENSIONS):
            os.remove(os.path.join(INPUT_DIR, f))
    for upload in input_files:
        src = upload if isinstance(upload, str) else upload.name
        dest = os.path.join(INPUT_DIR, os.path.basename(src))
        shutil.copy2(src, dest)
```

- [ ] **Step 4: Replace `process()` (lines 85–112)**

Replace the existing `process()` function with:

```python
def process(input_files, progress=gr.Progress()):
    if not input_files:
        yield "No files uploaded.", [], 0, 0, 0
        return

    _clear_and_copy(input_files)

    total = count_images(INPUT_DIR)
    if total == 0:
        yield "No supported images found in upload.", [], 0, 0, 0
        return

    results = []
    for img_result in run(INPUT_DIR, OUTPUT_DIR, MODELS):
        results.append(img_result)
        progress(len(results) / total)
        yield (
            _build_log(results),
            _list_output_images(),
            len(results),
            sum(r.faces_swapped for r in results),
            _warning_count(results),
        )

    stats = RunStats.from_results(total, results)
    yield (
        _build_log(results) + "\n" + str(stats),
        _list_output_images(),
        total,
        stats.faces_swapped,
        _warning_count(results),
    )


def clear_inputs():
    return None, "", [], 0, 0, 0


def open_output_folder():
    if sys.platform == "darwin":
        subprocess.run(["open", OUTPUT_DIR], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", OUTPUT_DIR], check=False)
    else:
        os.startfile(OUTPUT_DIR)
```

- [ ] **Step 5: Replace the Gradio layout block (lines 118–164)**

Replace the entire `with gr.Blocks(...) as demo:` block with:

```python
with gr.Blocks(title="Refacer", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
# Refacer
**Batch face anonymization for photojournalists and activist photographers.**
All processing happens locally — your images never leave this machine.
        """
    )
    gr.Markdown(_EXIFTOOL_NOTE)

    # ── Panel 1: Upload ────────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 1 · Upload")
        file_input = gr.File(
            label="Input images",
            file_count="multiple",
            file_types=list(SUPPORTED_EXTENSIONS),
        )
        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")
            clear_btn = gr.Button("Clear")

    # ── Panel 2: Progress ──────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 2 · Progress")
        log_output = gr.Textbox(
            label="Processing log",
            lines=10,
            interactive=False,
            autoscroll=True,
        )
        with gr.Row():
            done_num = gr.Number(label="Done", value=0, interactive=False)
            faces_num = gr.Number(label="Faces Swapped", value=0, interactive=False)
            warnings_num = gr.Number(label="Warnings", value=0, interactive=False)

    # ── Panel 3: Output ────────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### 3 · Output")
        gallery = gr.Gallery(
            label="Output images",
            columns=4,
            object_fit="contain",
            height="auto",
        )
        open_btn = gr.Button("Open output folder")

    run_btn.click(
        fn=process,
        inputs=[file_input],
        outputs=[log_output, gallery, done_num, faces_num, warnings_num],
    )
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[file_input, log_output, gallery, done_num, faces_num, warnings_num],
    )
    open_btn.click(fn=open_output_folder, inputs=[], outputs=[])
```

- [ ] **Step 6: Start the UI and verify manually**

```bash
python ui/app.py
```

Open `http://127.0.0.1:7860` and verify:

1. Three panels are visible stacked vertically: Upload, Progress, Output.
2. Upload a handful of images and click **Run**.
3. The log stream updates after each image completes (not all at once at the end).
4. **Done**, **Faces Swapped**, and **Warnings** numbers increment as images finish.
5. Gallery thumbnails appear as each image is written to `output/`.
6. The final log entry shows the `RunStats` aggregate summary.
7. **Clear** resets the file input, log, gallery, and stat numbers to empty/zero.
8. **Open output folder** opens the `output/` directory in Finder/Files.

- [ ] **Step 7: Commit**

```bash
git add ui/app.py
git commit -m "feat: rewrite UI as 3-panel streaming layout"
```

---

## Self-Review Notes

**Spec coverage check:**
- `run()` → generator ✓ Task 4
- `count_images()` ✓ Task 3
- `RunStats.from_results()` ✓ Task 2
- CLI adapter ✓ Task 5
- Three-panel layout (Upload / Progress / Output) ✓ Task 6
- `process()` generator with live log, gallery, and stat updates ✓ Task 6
- `_warning_count` consistent definition (images-with-warnings) ✓ Task 6
- `_clear_and_copy` extracted and named ✓ Task 6
- Cross-platform open-folder ✓ Task 6

**Type consistency check:**
- `RunStats.from_results(total, results)` — called identically in Task 2 (implementation), Task 5 (CLI), and Task 6 (UI). ✓
- `count_images(input_dir)` — defined in Task 3, called in Task 6. ✓
- `_warning_count(results)` — defined and called only in Task 6. ✓
- `run()` yields `ImageResult` — consumed in Task 4 tests, Task 5 CLI, Task 6 UI. ✓
