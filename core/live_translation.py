"""Capture desktop frames and translate the newest frame on a separate thread."""

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from math import isfinite
from threading import Condition, Thread
from time import perf_counter, sleep
from typing import Callable, Protocol
from unicodedata import normalize

import numpy as np
from numpy.typing import NDArray


Region = tuple[int, int, int, int]  # x, y, width, height in screen pixels
Box = tuple[int, int, int, int]  # left, top, right, bottom, relative to region
Frame = NDArray[np.uint8]
_CACHE_SIZE = 1024
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowCrop:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_selection(cls, selection: Region, window: Region):
        x, y, width, height = selection
        window_x, window_y, window_width, window_height = window
        return cls(
            (x - window_x) / window_width,
            (y - window_y) / window_height,
            (x + width - window_x) / window_width,
            (y + height - window_y) / window_height,
        )

    def box(self, width: int, height: int) -> Box:
        left = min(width - 1, round(self.left * width))
        top = min(height - 1, round(self.top * height))
        return left, top, max(left + 1, round(self.right * width)), max(top + 1, round(self.bottom * height))


class _Models(Protocol):
    def ocr(self, frame: Frame) -> list[tuple[Box, str]]: ...

    def translate(self, texts: list[str]) -> list[str]: ...

    def close(self) -> None: ...


class _Capture(Protocol):
    def grab(self, region: Region) -> Frame | None:
        """Return a stable BGR frame, or None if no new frame is ready."""
        ...


@dataclass(frozen=True)
class TranslatedLine:
    box: Box
    source: str
    translation: str


@dataclass(frozen=True)
class LiveUpdate:
    region: Region
    lines: tuple[TranslatedLine, ...]
    error: str | None = None


class _MssCapture:
    def __init__(self, desktop):
        self._desktop = desktop

    def grab(self, region: Region) -> Frame:
        left, top, width, height = region
        pixels = self._desktop.grab(dict(left=left, top=top, width=width, height=height))
        return np.ascontiguousarray(np.asarray(pixels)[:, :, :3])


@contextmanager
def _open_desktop():
    from mss import mss

    with mss() as desktop:
        yield _MssCapture(desktop)


class LiveTranslationSession:
    """A latest-frame pipeline. Only ``poll()`` delivers updates to the UI."""

    def __init__(
        self,
        model_factory: Callable[[], _Models],
        *,
        capture_fps: float = 30,
        capture_factory: Callable[[], AbstractContextManager[_Capture]] | None = None,
        frame_crop: WindowCrop | None = None,
    ):
        if not isfinite(capture_fps) or capture_fps <= 0:
            raise ValueError('capture_fps must be finite and positive')
        self._interval = 1 / capture_fps
        self._model_factory = model_factory
        self._capture_factory = capture_factory if capture_factory is not None else _open_desktop
        self._frame_crop = frame_crop
        self._state = Condition()
        self._closed = False
        self._generation = 0
        self._region: Region | None = None
        self._last_captured: Frame | None = None
        self._pending_frame: Frame | None = None
        self._update: LiveUpdate | None = None
        self._models: _Models | None = None

        Thread(target=self._capture, name='live-capture', daemon=True).start()
        Thread(target=self._analyse, name='live-inference', daemon=True).start()

    def start(self, region: Region) -> None:
        if region[2] <= 0 or region[3] <= 0:
            raise ValueError('Live translation needs a positive region width and height')
        with self._state:
            if not self._closed:
                self._set_region(region)

    def stop(self) -> None:
        with self._state:
            self._set_region(None)

    def clear(self) -> None:
        with self._state:
            if self._region is not None:
                self._generation += 1
                self._pending_frame = None
                self._update = LiveUpdate(self._region, ())

    def poll(self) -> LiveUpdate | None:
        with self._state:
            update, self._update = self._update, None
            return update

    def close(self) -> None:
        with self._state:
            self._closed = True
            self._set_region(None)
            models, self._models = self._models, None
        if models is not None:
            # Close can interrupt a model subprocess while it is serving an inference request.
            Thread(target=self._close_models, args=(models,), name='live-model-close', daemon=False).start()

    def _set_region(self, region: Region | None) -> None:
        # All shared state changes here occur under the condition lock.
        self._generation += 1
        self._region = region
        self._last_captured = None
        self._pending_frame = None
        self._update = None
        self._state.notify_all()

    def _fail(self, generation: int, region: Region, operation: str, error: Exception) -> None:
        with self._state:
            if generation == self._generation:
                self._set_region(None)
                self._update = LiveUpdate(region, (), f'{operation}: {str(error) or type(error).__name__}')

    def _capture(self) -> None:
        while True:
            with self._state:
                self._state.wait_for(lambda: self._closed or self._region is not None)
                region = self._region
                generation = self._generation
                if region is None:
                    return
            try:
                with self._capture_factory() as capture:
                    self._capture_frames(capture, region)
            except Exception as error:
                self._fail(generation, region, 'Screen capture failed', error)

    def _capture_frames(self, capture: _Capture, region: Region) -> None:
        box = self._frame_crop.box(*region[2:]) if self._frame_crop else None
        while True:
            with self._state:
                if self._region != region:
                    return
                generation = self._generation
                previous = self._last_captured

            deadline = perf_counter() + self._interval
            try:
                frame = capture.grab(region)
                if frame is not None and box is not None:
                    left, top, right, bottom = box
                    frame = np.ascontiguousarray(frame[top:bottom, left:right])
                changed = frame is not None and (previous is None or not np.array_equal(previous, frame))
            except Exception as error:
                self._fail(generation, region, 'Screen capture failed', error)
                return

            with self._state:
                if changed and generation == self._generation:
                    self._last_captured = frame
                    self._pending_frame = frame
                    self._state.notify_all()

            # On Windows, sleep uses a high-resolution timer. A condition timeout
            # rounds up to a system clock tick and can miss the 60 FPS interval.
            remaining = deadline - perf_counter()
            if remaining > 0:
                sleep(remaining)

    def _analyse(self) -> None:
        cache: OrderedDict[str, str] = OrderedDict()
        last_frame = None
        last_generation = None
        while True:
            with self._state:
                self._state.wait_for(lambda: self._closed or self._pending_frame is not None)
                if self._closed:
                    return
                frame, self._pending_frame = self._pending_frame, None
                region = self._region
                generation = self._generation
                models = self._models

            if generation == last_generation and np.array_equal(last_frame, frame):
                continue

            try:
                if models is None:
                    models = self._model_factory()
                    with self._state:
                        closed = self._closed
                        if not closed:
                            self._models = models
                    if closed:
                        self._close_models(models)
                        return
                    with self._state:
                        if generation != self._generation:
                            continue

                detected = models.ocr(frame)
                if self._frame_crop:
                    left, top, _, _ = self._frame_crop.box(*region[2:])
                    detected = [
                        ((x1 + left, y1 + top, x2 + left, y2 + top), text)
                        for (x1, y1, x2, y2), text in detected
                    ]
                with self._state:
                    if generation != self._generation:
                        continue
                lines = self._translate(models, detected, cache)
            except Exception as error:
                self._fail(generation, region, 'Live inference failed', error)
                with self._state:
                    if models is None or self._models is not models:
                        continue
                    self._models = None
                self._close_models(models)
                continue

            last_frame, last_generation = frame, generation
            with self._state:
                if generation == self._generation:
                    self._update = LiveUpdate(region, lines)

    @staticmethod
    def _translate(models: _Models, detected: list[tuple[Box, str]], cache: OrderedDict[str, str]):
        entries = []
        translations = {}
        missing = {}
        for box, source in detected:
            text = ' '.join(normalize('NFC', source).split())
            if not any(character.isalpha() for character in text):
                continue
            entries.append((box, source, text))
            if text in cache:
                cache.move_to_end(text)
                translations[text] = cache[text]
            else:
                missing[text] = None

        if missing:
            results = models.translate(list(missing))
            for text, translation in zip(missing, results, strict=True):
                translations[text] = translation
            for text in missing:
                cache[text] = translations[text]
                if len(cache) > _CACHE_SIZE:
                    cache.popitem(last=False)

        return tuple(
            TranslatedLine(box, source, translations[text])
            for box, source, text in entries
            if translations[text].strip()
        )

    @staticmethod
    def _close_models(models: _Models) -> None:
        try:
            models.close()
        except Exception:
            _LOG.exception('Error closing live translation models')
