"""Capture desktop frames and translate the newest frame on a separate thread."""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
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
_DIALOGUE_EXTENSION_SECONDS = 3
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

    def translate_contextual(self, text: str, *, role: str, rules: str, speaker: str, history) -> str: ...

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
    token_count: int | None = None
    request_id: int | None = None
    render_id: int | None = None


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


def _group_dialogue_lines(detected: list[tuple[Box, str]]) -> list[tuple[Box, str]]:
    """Keep neighboring subtitle lines together without combining separate columns."""
    groups = []
    for box, text in sorted(detected, key=lambda item: (item[0][1], item[0][0])):
        left, top, right, bottom = box
        height = bottom - top
        for group in reversed(groups):
            previous, _ = group[-1]
            other_left, other_top, other_right, other_bottom = previous
            other_height = other_bottom - other_top
            overlap = min(right, other_right) - max(left, other_left)
            if (
                len(group) < 3
                and -0.2 * height <= top - other_bottom <= 0.55 * min(height, other_height)
                and max(height, other_height) <= 1.5 * min(height, other_height)
                and overlap >= 0.5 * min(right - left, other_right - other_left)
            ):
                group.append((box, text))
                break
        else:
            groups.append([(box, text)])

    return [
        (
            (min(box[0] for box, _ in group), group[0][0][1],
             max(box[2] for box, _ in group), max(box[3] for box, _ in group)),
            "\n".join(text for _, text in group),
        )
        for group in groups
    ]


@dataclass(frozen=True)
class _DialogueTurn:
    crop_index: int
    box: Box
    updated_at: float
    context: tuple[str, str, str]

    def continues(self, crop_index, box, speaker, text, recent, now):
        old_speaker, old_text, old_translation = self.context
        if (
            crop_index != self.crop_index or speaker != old_speaker
            or recent.get(old_text) != old_translation
            or now - self.updated_at > _DIALOGUE_EXTENSION_SECONDS
        ):
            return False

        old_text = " ".join(old_text.split())
        text = " ".join(text.split())
        suffix = text[len(old_text):]
        if not suffix or not text.startswith(old_text):
            return False
        sentence = old_text.rstrip(" \"'”’」』）)]】〉》")
        if sentence.endswith((".", "!", "?", "。", "！", "？", "…")) and any(
            character.isalnum() for character in suffix
        ):
            return False

        left, top, right, bottom = box
        old_left, old_top, old_right, old_bottom = self.box
        overlap = min(right, old_right) - max(left, old_left)
        return (
            overlap >= .5 * min(right - left, old_right - old_left)
            and abs(top - old_top) <= .5 * min(bottom - top, old_bottom - old_top)
        )


class LiveTranslationSession:
    """A latest-frame pipeline. Only ``poll()`` delivers updates to the UI."""

    def __init__(
        self,
        model_factory: Callable[[], _Models],
        *,
        capture_fps: float = 30,
        capture_factory: Callable[[], AbstractContextManager[_Capture]] | None = None,
        frame_crop: WindowCrop | None = None,
        group_dialogue: bool = False,
        frame_crops: tuple[tuple[WindowCrop, str, str], ...] = (),
        history_turns: int = 0,
        once_timeout: float = 5,
        post_translation_delay_ms: int = 0,
    ):
        if not isfinite(capture_fps) or capture_fps <= 0:
            raise ValueError('capture_fps must be finite and positive')
        self._interval = 1 / capture_fps
        if not isfinite(once_timeout) or once_timeout <= 0:
            raise ValueError('once_timeout must be finite and positive')
        self._once_timeout = once_timeout
        self._model_factory = model_factory
        self._capture_factory = capture_factory if capture_factory is not None else _open_desktop
        self._frame_crop = frame_crop
        self._group_dialogue = group_dialogue
        self._frame_crops = tuple(frame_crops)
        self._history_turns = history_turns
        self._post_translation_delay_ms = max(0, post_translation_delay_ms)
        self._render_serial = 0
        self._render_gate = None
        self._suppress_unchanged = False
        self._context_revision = 0
        self._live_cache_revision = 0
        self._state = Condition()
        self._closed = False
        self._generation = 0
        self._region: Region | None = None
        self._once_id: int | None = None
        self._once_needs_frame = False
        self._last_captured: Frame | None = None
        self._completed_context_frame: Frame | None = None
        self._pending_frame: Frame | None = None
        self._update: LiveUpdate | None = None
        self._models: _Models | None = None

        Thread(target=self._capture, name='live-capture', daemon=True).start()
        Thread(target=self._analyse, name='live-inference', daemon=True).start()

    def start(self, region: Region, *, reset_context=True) -> None:
        if region[2] <= 0 or region[3] <= 0:
            raise ValueError('Live translation needs a positive region width and height')
        with self._state:
            if not self._closed:
                self._set_region(region, reset_context=reset_context)

    def request_once(self, region: Region) -> int | None:
        """Queue one frame and return its completion ID, or None if busy."""
        if region[2] <= 0 or region[3] <= 0:
            raise ValueError('Live translation needs a positive region width and height')
        with self._state:
            if self._closed or self._region is not None:
                return None
            self._set_region(region, reset_context=False)
            self._once_id = self._generation
            self._once_needs_frame = True
            return self._once_id

    def move(self, region: Region) -> None:
        with self._state:
            if self._region is not None:
                if region[2:] != self._region[2:]:
                    raise ValueError("Moving live capture requires the same region size")
                self._region = region
            if self._update is not None:
                self._update = replace(self._update, region=region)

    def stop(self, *, reset_context=True) -> None:
        with self._state:
            self._set_region(None, reset_context=reset_context)

    def configure_live(self, *, frame_crop, frame_crops, history_turns, post_translation_delay_ms) -> None:
        with self._state:
            frame_crops = tuple(frame_crops)
            delay = max(0, post_translation_delay_ms)
            crop_changed = (frame_crop, frame_crops) != (self._frame_crop, self._frame_crops)
            if not crop_changed and (history_turns, delay) == (self._history_turns, self._post_translation_delay_ms):
                return
            delay_only = not crop_changed and history_turns == self._history_turns
            self._frame_crop, self._frame_crops = frame_crop, frame_crops
            self._history_turns, self._post_translation_delay_ms = history_turns, delay
            if delay_only:
                self._render_gate = None
                self._state.notify_all()
                return
            self._generation += 1
            if crop_changed:
                self._context_revision += 1
                self._live_cache_revision = self._context_revision
                self._suppress_unchanged = False
            preserve_once = not crop_changed and self._update is not None and self._update.request_id is not None
            if self._once_id is None:
                self._last_captured = None
            # An active one-shot reuses its snapshot under the new settings.
            self._pending_frame = self._last_captured
            self._completed_context_frame = None
            if not preserve_once:
                self._update = None
            self._render_gate = None
            self._state.notify_all()

    def acknowledge_render(self, render_id: int) -> None:
        with self._state:
            if (self._render_gate is not None and self._render_gate[0] == render_id
                    and self._render_gate[1] is None):
                self._render_gate = (render_id, perf_counter() + self._post_translation_delay_ms / 1000)
                self._state.notify_all()

    def _publish_live(self, lines, token_count):
        render_id = None
        if lines and self._post_translation_delay_ms > 0:
            self._render_serial += 1
            render_id = self._render_serial
            self._render_gate = (render_id, None)
        self._update = LiveUpdate(self._region, lines, token_count=token_count, render_id=render_id)

    def clear(self) -> None:
        with self._state:
            if self._region is not None and self._once_id is None:
                self._generation += 1
                self._render_gate = None
                self._suppress_unchanged = True
                self._state.notify_all()
                self._pending_frame = None
                self._update = LiveUpdate(self._region, ())

    def clear_context(self) -> None:
        with self._state:
            self._context_revision += 1
            if self._region is None or not (self._frame_crops or self._history_turns):
                return

            if self._once_id is not None:
                # Keep the snapshot and request ID even if the screen has changed.
                self._pending_frame = self._last_captured
            else:
                if self._update is not None and not self._suppress_unchanged:
                    self._update = None
                    self._render_gate = None
                    self._completed_context_frame = None
                    self._live_cache_revision = self._context_revision
                # Leave completed, displayed dialogue on screen without adding it
                # back to history. Replay only a frame with unfinished work.
                if self._last_captured is not self._completed_context_frame:
                    self._pending_frame = self._last_captured
            self._state.notify_all()

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

    def _set_region(self, region: Region | None, *, reset_context=True) -> None:
        # All shared state changes here occur under the condition lock.
        self._generation += 1
        self._render_gate = None
        if reset_context:
            self._context_revision += 1
            self._live_cache_revision = self._context_revision
            self._suppress_unchanged = False
        self._region = region
        self._once_id = None
        self._once_needs_frame = False
        self._last_captured = None
        self._completed_context_frame = None
        self._pending_frame = None
        self._update = None
        self._state.notify_all()

    def _fail(self, generation: int, operation: str, error: Exception, *, revision: int | None = None) -> None:
        with self._state:
            if self._analysis_is_current((generation, revision)) and self._region is not None:
                update = LiveUpdate(
                    self._region, (), f'{operation}: {str(error) or type(error).__name__}',
                    request_id=self._once_id,
                )
                self._set_region(None, reset_context=self._once_id is None)
                self._update = update

    def _capture(self) -> None:
        while True:
            with self._state:
                self._state.wait_for(lambda: self._closed or (
                    self._region is not None and (self._once_id is None or self._once_needs_frame)
                ))
                region = self._region
                generation = self._generation
                request_id = self._once_id
                if region is None:
                    return
            try:
                with self._capture_factory() as capture:
                    if request_id is not None:
                        frame = self._capture_once(capture, region, request_id)
                    else:
                        self._capture_frames(capture, region)
                if request_id is not None and frame is not None:
                    with self._state:
                        if request_id == self._once_id:
                            self._once_needs_frame = False
                            self._last_captured = self._pending_frame = frame
                            self._state.notify_all()
            except Exception as error:
                with self._state:
                    if request_id is not None:
                        if request_id != self._once_id:
                            continue
                        generation = self._generation
                    self._fail(generation, 'Screen capture failed', error)

    def _capture_crop_box(self, size):
        if self._frame_crop and not (self._frame_crops or self._history_turns):
            return self._frame_crop.box(*size)
        return None

    @staticmethod
    def _crop_frame(frame, box):
        left, top, right, bottom = box
        return np.ascontiguousarray(frame[top:bottom, left:right])

    def _capture_once(self, capture: _Capture, region: Region, request_id: int) -> Frame | None:
        timeout = perf_counter() + self._once_timeout
        while True:
            with self._state:
                if request_id != self._once_id:
                    return None
            frame = capture.grab(region)
            if frame is not None and frame.shape[:2] == (region[3], region[2]):
                # Keep the full snapshot so analysis can crop it using the current settings.
                return frame
            remaining = timeout - perf_counter()
            if remaining <= 0:
                raise TimeoutError('Timed out waiting for a matching frame from the capture source.')
            sleep(min(self._interval, remaining))

    def _capture_frames(self, capture: _Capture, region: Region) -> None:
        size = region[2:]
        while True:
            with self._state:
                region = self._region
                if region is None or region[2:] != size or self._once_id is not None:
                    return
                generation = self._generation
                previous = self._last_captured
                box = self._capture_crop_box(size)

            deadline = perf_counter() + self._interval
            try:
                frame = capture.grab(region)
                if frame is not None and box is not None:
                    frame = self._crop_frame(frame, box)
                changed = frame is not None and (previous is None or not np.array_equal(previous, frame))
            except Exception as error:
                self._fail(generation, 'Screen capture failed', error)
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

    def _analysis_is_current(self, key: tuple[int, int | None]) -> bool:
        """Check the captured generation and, for contextual work, its history revision under lock."""
        generation, revision = key
        return generation == self._generation and (revision is None or revision == self._context_revision)

    def _analyse(self) -> None:
        cache: OrderedDict[str, str] = OrderedDict()
        history = deque(maxlen=self._history_turns)
        region_lines = []
        previous_tiles = []
        speakers = []
        context_revision = 0
        last_frame = None
        last_analysis = None
        last_token_count = None
        previous_once = None
        committed_live = None
        while True:
            with self._state:
                while not self._closed:
                    if self._render_gate is not None:
                        deadline = self._render_gate[1]
                        if deadline is None:
                            self._state.wait()
                            continue
                        remaining = deadline - perf_counter()
                        if remaining > 0:
                            self._state.wait(remaining)
                            continue
                        self._render_gate = None
                    if self._pending_frame is not None:
                        break
                    self._state.wait()
                if self._closed:
                    return
                frame, self._pending_frame = self._pending_frame, None
                region = self._region
                generation = self._generation
                models = self._models
                request_id = self._once_id
                revision = self._context_revision
                live_cache_revision = self._live_cache_revision
                frame_crop, frame_crops = self._frame_crop, self._frame_crops
                history_turns = self._history_turns
                suppress_unchanged = self._suppress_unchanged
                contextual = bool(frame_crops or history_turns)
                analysis = generation, revision if contextual else None

            if request_id is None and suppress_unchanged:
                same_frame = last_frame is not None and np.array_equal(last_frame, frame)
                if (not same_frame and last_frame is not None and frame_crops
                        and frame.shape == last_frame.shape):
                    height, width = frame.shape[:2]
                    same_frame = all(
                        np.array_equal(
                            self._crop_frame(last_frame, crop.box(width, height)) if crop else last_frame,
                            self._crop_frame(frame, crop.box(width, height)) if crop else frame,
                        )
                        for crop, _, _ in frame_crops
                    )
                if not same_frame and last_frame is not None and frame_crop and not frame_crops:
                    box = frame_crop.box(*region[2:])
                    full_shape = (region[3], region[2])
                    if frame.shape[:2] == full_shape and last_frame.shape[:2] == full_shape:
                        same_frame = np.array_equal(self._crop_frame(last_frame, box), self._crop_frame(frame, box))
                    elif frame.shape[:2] == full_shape:
                        same_frame = np.array_equal(last_frame, self._crop_frame(frame, box))
                    elif last_frame.shape[:2] == full_shape:
                        same_frame = np.array_equal(self._crop_frame(last_frame, box), frame)
                if same_frame:
                    continue
                with self._state:
                    if not self._analysis_is_current(analysis):
                        continue
                    self._suppress_unchanged = False

            if history.maxlen != history_turns:
                history = deque(history, maxlen=history_turns)
            if context_revision != revision:
                history.clear()
                context_revision = revision

            if analysis == last_analysis and np.array_equal(last_frame, frame):
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
                        if not self._analysis_is_current(analysis):
                            continue

                if contextual:
                    crops = frame_crops or ((frame_crop, "dialogue", ""),)
                    once_key = revision, region[2:]
                    live_key = live_cache_revision, region[2:], crops
                    if request_id is None or (previous_once and previous_once[0] != once_key):
                        previous_once = None
                    # Only commit context from a completed, uncancelled round.
                    round_history = history.copy()
                    generation_changed = last_analysis is None or generation != last_analysis[0]
                    analysis_changed = analysis != last_analysis
                    if analysis_changed:
                        if previous_once is not None:
                            _, region_lines, previous_tiles, speakers, last_token_count = previous_once
                            region_lines, previous_tiles, speakers = (
                                region_lines.copy(), previous_tiles.copy(), speakers.copy()
                            )
                        elif request_id is None and committed_live is not None and committed_live[0] == live_key:
                            _, region_lines, previous_tiles, speakers, last_token_count = committed_live
                            region_lines, previous_tiles, speakers = (
                                region_lines.copy(), previous_tiles.copy(), speakers.copy()
                            )
                        else:
                            region_lines = [()] * len(crops)
                            previous_tiles = [None] * len(crops)
                            speakers = [""] * len(crops)
                            last_token_count = None
                    last_frame, last_analysis = frame, analysis
                    speaker = ""
                    changed_batch = False
                    for index, (crop, role, rules) in enumerate(crops):
                        with self._state:
                            if not self._analysis_is_current(analysis):
                                break
                        height, width = frame.shape[:2]
                        left, top, right, bottom = crop.box(width, height) if crop else (0, 0, width, height)
                        tile = frame[top:bottom, left:right]
                        incoming_speaker = speaker if role == "dialogue" else ""
                        same_speaker = speakers[index] == incoming_speaker
                        if not same_speaker or not np.array_equal(previous_tiles[index], tile):
                            lines, tokens = self._translate_crop(
                                models, tile, index, role, rules, incoming_speaker, round_history,
                                region_lines[index] if same_speaker else (), (left, top), cache, analysis,
                            )
                            changed_batch = True
                            previous_tiles[index] = tile
                            speakers[index] = incoming_speaker
                            region_lines[index] = lines
                            if tokens is not None:
                                last_token_count = tokens
                            with self._state:
                                if not self._analysis_is_current(analysis):
                                    break
                                if request_id is None and self._post_translation_delay_ms == 0:
                                    self._update = LiveUpdate(
                                        self._region, tuple(line for rows in region_lines for line in rows),
                                        token_count=last_token_count,
                                    )
                        if role == "speaker":
                            speaker = " ".join(line.translation for line in region_lines[index])
                    if request_id is None:
                        with self._state:
                            if self._analysis_is_current(analysis):
                                history = round_history
                                self._completed_context_frame = frame
                                committed_live = (live_key, region_lines.copy(), previous_tiles.copy(), speakers.copy(), last_token_count)
                                if (changed_batch and self._post_translation_delay_ms > 0) or (generation_changed and not changed_batch):
                                    self._publish_live(tuple(line for rows in region_lines for line in rows), last_token_count)
                        continue
                    lines = tuple(line for rows in region_lines for line in rows)
                    token_count = last_token_count

                else:
                    crop_box = frame_crop.box(*region[2:]) if frame_crop else None
                    ocr_frame = self._crop_frame(frame, crop_box) if request_id is not None and crop_box else frame
                    detected = models.ocr(ocr_frame)
                    if crop_box:
                        left, top, _, _ = crop_box
                        detected = [
                            ((x1 + left, y1 + top, x2 + left, y2 + top), text)
                            for (x1, y1, x2, y2), text in detected
                        ]
                    with self._state:
                        if not self._analysis_is_current(analysis):
                            continue
                    if self._group_dialogue:
                        detected = _group_dialogue_lines(detected)
                    lines, token_count = self._translate(
                        models, detected, cache, preserve_line_breaks=self._group_dialogue,
                    )
            except Exception as error:
                self._fail(generation, 'Live inference failed', error, revision=analysis[1])
                with self._state:
                    if models is None or self._models is not models:
                        continue
                    self._models = None
                self._close_models(models)
                continue

            with self._state:
                if self._analysis_is_current(analysis):
                    if token_count is not None or analysis != last_analysis:
                        last_token_count = token_count
                    last_frame, last_analysis = frame, analysis
                    update = LiveUpdate(self._region, lines, token_count=last_token_count, request_id=request_id)
                    if request_id is not None:
                        if contextual:
                            history = round_history
                            previous_once = once_key, region_lines, previous_tiles, speakers, last_token_count
                        self._set_region(None, reset_context=False)
                    if request_id is None:
                        self._publish_live(lines, last_token_count)
                    else:
                        self._update = update

    def _translate_crop(self, models, tile, crop_index, role, rules, speaker, history, previous, offset, cache, analysis):
        detected = models.ocr(np.ascontiguousarray(tile))
        if self._group_dialogue and role == "dialogue":
            detected = _group_dialogue_lines(detected)
        else:
            detected.sort(key=lambda item: (item[0][1], item[0][0]))
        left, top = offset
        detected = [((x1 + left, y1 + top, x2 + left, y2 + top), source)
                    for (x1, y1, x2, y2), source in detected]
        if not self._group_dialogue and role == "dialogue":
            return self._translate(models, detected, cache)

        recent = {self._normalize(line.source, True): line.translation for line in previous}
        lines = []
        token_count = None
        for box, source in detected:
            text = self._normalize(source, True)
            if not text:
                continue
            translation = recent.get(text)
            if translation is None:
                draft_index = None
                if self._group_dialogue:
                    with self._state:
                        if not self._analysis_is_current(analysis):
                            return (), None
                    if role == "dialogue" and recent:
                        now = perf_counter()
                        draft_index = next((
                            index for index in reversed(range(len(history)))
                            if history[index].continues(crop_index, box, speaker, text, recent, now)
                        ), None)
                    translation = models.translate_contextual(
                        text, role=role, rules=rules, speaker=speaker,
                        history=tuple(turn.context for index, turn in enumerate(history) if index != draft_index),
                    )
                    count = getattr(models, "last_token_count", None)
                    if count is not None:
                        token_count = count
                else:
                    translation = text
                if translation.strip() and role == "dialogue":
                    turn = _DialogueTurn(crop_index, box, perf_counter(), (speaker, text, translation))
                    if draft_index is None:
                        history.append(turn)
                    else:
                        history[draft_index] = turn
            if translation.strip():
                lines.append(TranslatedLine(box, source, translation))
        return tuple(lines), token_count

    @staticmethod
    def _normalize(source, preserve_line_breaks):
        normalized = normalize("NFC", source)
        if preserve_line_breaks:
            text = "\n".join(" ".join(line.split()) for line in normalized.splitlines() if line.strip())
        else:
            text = " ".join(normalized.split())
        return text if any(character.isalpha() for character in text) and sum(
            character.isalnum() for character in text
        ) >= 2 else ""

    @staticmethod
    def _translate(
        models: _Models,
        detected: list[tuple[Box, str]],
        cache: OrderedDict[str, str],
        *,
        preserve_line_breaks: bool = False,
    ):
        entries = []
        translations = {}
        missing = {}
        for box, source in detected:
            text = LiveTranslationSession._normalize(source, preserve_line_breaks)
            if not text:
                continue
            entries.append((box, source, text))
            if text in cache:
                cache.move_to_end(text)
                translations[text] = cache[text]
            else:
                missing[text] = None

        token_count = None
        if missing:
            results = models.translate(list(missing))
            token_count = getattr(models, 'last_token_count', None)
            for text, translation in zip(missing, results, strict=True):
                translations[text] = translation
            for text in missing:
                cache[text] = translations[text]
                if len(cache) > _CACHE_SIZE:
                    cache.popitem(last=False)

        lines = tuple(
            TranslatedLine(box, source, translations[text])
            for box, source, text in entries
            if translations[text].strip()
        )
        return lines, token_count

    @staticmethod
    def _close_models(models: _Models) -> None:
        try:
            models.close()
        except Exception:
            _LOG.exception('Error closing live translation models')
