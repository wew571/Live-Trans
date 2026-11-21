# Live Translation Implementation Plan

> For agentic workers, implement via subagent-driven development. Work only in your assigned files and preserve the existing staged and unstaged changes.

**Goal:** Add a MangaLens mode for continuous English-to-Vietnamese overlay translation of the full main screen or the user's selected area.

**Architecture:** A Windows capture thread publishes the newest screen crop at a configurable 30 or 60 FPS. A separate analysis thread runs PP-OCRv6 Small and translates new lines in batches. The UI consumes completed snapshots on its main thread, while an isolated model process keeps the newer model dependencies separate.

**Tech Stack:** Python, CustomTkinter, MSS, PaddleOCR 3.7, Transformers, PyTorch CUDA, and the existing VinAI English-to-Vietnamese model.

**Spec:** User-approved mode and model selection in the MangaLens chat. The selected-region benchmark informs refresh expectations.

## Global Constraints

- Keep the existing Manga mode, controls, and hotkeys functional.
- Use PP-OCRv6 Small for detection and recognition. Default to 30 capture FPS and accept 60 via configuration.
- Default translator is VinAI. Its input and output are batches of strings, independent of capture and overlay.
- Keep inference local. Start models lazily. Never publish callbacks into Tk from a model or capture thread.
- Mode switch, stop, region change, and shutdown must invalidate pending overlay updates.
- Preserve preexisting workspace changes and keep feature work unstaged.

## Task 1. Capture and session

- [x] Add failing unit tests for latest-frame coalescing, unchanged/cached text, region changes, stale completion, and error reporting.
- [x] Implement core/live_translation.py. LiveTranslationSession.start(region), stop(), poll(), clear(), and close() back the UI. Capture publishes BGR images and local OCR coordinates are carried in the result.
- [x] Run the tests and review stop/restart, latest-only coalescing and fake capture failures.

## Task 2. Inference and replaceable translator

- [x] Add a failing unit test for the text translator contract and model-client request validation.
- [x] Implement core/live_models.py and its worker. LiveModelClient exposes ocr(frame), translate(texts), and close(). Use the benchmarked PP-OCRv6 Small detector/recognizer and English-to-Vietnamese VinAI translation.
- [x] Add isolated runtime setup and verify a real frame through both model calls.

## Task 3. Mode and overlay

- [x] Exercise control state and overlay coordinate conversion through focused tests.
- [x] Implement mode routing in frontend/ui.py, frontend/live_overlay.py and the existing Manga hotkeys where needed. Enable and disable live mode with the normal scan/monitor controls, and support start/stop, clear, region selection, and shutdown.
- [x] Verify the live overlay is click-through, is excluded from captures, and anchors to the selected region. Check round-trip mode switching.

## Task 4. Integration

- [x] Provision the model runtime on the current machine and document how to set it up and change translators.
- [x] Run feature tests, real OCR and translation, and GUI smoke checks. Identify measured scope and platform limitations in the handoff.
