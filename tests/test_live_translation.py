import threading
import time
import unittest
from collections import Counter, OrderedDict
from contextlib import contextmanager

import numpy as np
import pytest

from core import live_translation
from core.live_translation import LiveTranslationSession


REGION = (-100, 20, 4, 3)
OTHER_REGION = (50, 60, 4, 3)
BOX = (1, 0, 4, 3)


def test_ordered_crops_keep_speaker_and_recent_dialogue():
    from contextlib import nullcontext

    from core.live_translation import WindowCrop

    class Capture:
        dialogue = 2

        def grab(self, region):
            frame = np.ones((4, 8, 3), dtype=np.uint8)
            frame[:, 4:] = self.dialogue
            return frame

    class ContextualModels:
        def __init__(self):
            self.calls = []

        def ocr(self, frame):
            text = {1: "カレン", 2: "First line", 3: "Next line", 4: "New scene"}[int(frame[0, 0, 0])]
            return [((0, 0, 4, 4), text)]

        def translate_contextual(self, text, **context):
            self.calls.append((text, context))
            return "Curren" if context["role"] == "speaker" else "VI " + text

        def close(self):
            pass

    capture, models = Capture(), ContextualModels()
    region = (50, 60, 8, 4)
    session = LiveTranslationSession(
        lambda: models, capture_factory=lambda: nullcontext(capture),
        frame_crops=(
            (WindowCrop(0, 0, .5, 1), "speaker", "Names"),
            (WindowCrop(.5, 0, 1, 1), "dialogue", "Dialogue"),
        ),
        group_dialogue=True, history_turns=2,
    )

    def wait_translation(translation):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update and any(line.translation == translation for line in update.lines):
                return update.lines
            time.sleep(.01)
        raise AssertionError(f"Missing translation: {translation}")

    try:
        session.start(region)
        result = wait_translation("VI First line")
        assert [line.box for line in result] == [(0, 0, 4, 4), (4, 0, 8, 4)]
        capture.dialogue = 3
        wait_translation("VI Next line")
        assert [source for source, _ in models.calls] == ["カレン", "First line", "Next line"]
        assert models.calls[-1][1] == {
            "role": "dialogue", "rules": "Dialogue", "speaker": "Curren",
            "history": (("Curren", "First line", "VI First line"),),
        }
        session.clear_context()
        capture.dialogue = 4
        wait_translation("VI New scene")
        assert models.calls[-1][1]["history"] == ()
        assert models.calls[-1][1]["speaker"] == "Curren"
    finally:
        session.close()


class Screen:
    def __init__(self):
        self._lock = threading.Condition()
        self._value = 1
        self._counts = Counter()
        self._failure = None
        self.timestamps = []

    @contextmanager
    def open(self):
        yield self

    def grab(self, region):
        with self._lock:
            if self._failure:
                raise self._failure
            value = self._value
            self._counts[value] += 1
            self.timestamps.append(time.perf_counter())
            self._lock.notify_all()
        return np.full((region[3], region[2], 3), value, dtype=np.uint8)

    def show(self, value):
        with self._lock:
            self._value = value

    def wait_for_captures(self, value, count=2):
        with self._lock:
            return self._lock.wait_for(lambda: self._counts[value] >= count, timeout=3)

    def fail(self, error):
        with self._lock:
            self._failure = error


class Models:
    def __init__(self):
        self.ocr_values = []
        self.batches = []
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.closed = threading.Event()
        self.failure = None
        self.resume.set()
        self.text = {1: 'First line', 2: 'First   line', 3: 'Third line'}

    def ocr(self, frame):
        value = int(frame[0, 0, 0])
        self.ocr_values.append(value)
        self.entered.set()
        if not self.resume.wait(timeout=5):
            raise TimeoutError('test did not release OCR')
        if self.failure:
            raise self.failure
        return [(BOX, self.text[value])] if value in self.text else []

    def translate(self, texts):
        self.batches.append(texts)
        return [f'VI {text}' for text in texts]

    def close(self):
        self.closed.set()


class LiveTranslationTests(unittest.TestCase):
    def setUp(self):
        self.screen = Screen()
        self.models = Models()
        self.created = []

        def create_models():
            self.created.append(threading.get_ident())
            return self.models

        self.session = LiveTranslationSession(create_models, capture_factory=self.screen.open)
        self.addCleanup(self.session.close)
        self.addCleanup(self.models.resume.set)

    def wait_for_update(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = self.session.poll()
            if update is not None:
                return update
            time.sleep(0.005)
        self.fail('No live translation update arrived')

    def test_window_crop_limits_ocr_and_restores_window_relative_coordinates(self):
        region = (100, 200, 8, 6)
        crop = live_translation.WindowCrop.from_selection((102, 201, 4, 3), region)
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[1:4, 2:6] = 1

        @contextmanager
        def open_capture():
            yield type('Capture', (), {'grab': lambda self, region: frame})()

        session = LiveTranslationSession(lambda: self.models, capture_factory=open_capture, frame_crop=crop)
        self.addCleanup(session.close)
        session.start(region)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update is not None:
                self.assertEqual(update.lines[0].box, (3, 1, 6, 4))
                self.assertEqual(self.models.ocr_values, [1])
                return
            time.sleep(0.005)
        self.fail('No cropped translation update arrived')

    def test_reuses_warm_models_and_caches_normalized_text(self):
        self.assertEqual(self.created, [])
        self.session.start(REGION)
        first = self.wait_for_update()
        self.assertEqual(first.region, REGION)
        self.assertEqual(first.lines[0].box, BOX)
        self.assertEqual(first.lines[0].translation, 'VI First line')

        self.session.stop()
        self.screen.show(2)
        self.session.start(OTHER_REGION)
        second = self.wait_for_update()
        self.assertEqual(second.region, OTHER_REGION)
        self.assertEqual(second.lines[0].translation, 'VI First line')
        self.assertEqual(self.models.batches, [['First line']])
        self.assertEqual(len(self.created), 1)
        self.assertNotEqual(self.created[0], threading.get_ident())
        self.session.close()
        self.assertTrue(self.models.closed.wait(3))

    def test_capture_continues_during_ocr_and_keeps_only_latest_frame(self):
        self.models.resume.clear()
        self.session.start(REGION)
        self.assertTrue(self.models.entered.wait(3))
        self.screen.show(2)
        self.assertTrue(self.screen.wait_for_captures(2))
        self.screen.show(3)
        self.assertTrue(self.screen.wait_for_captures(3))
        self.models.resume.set()

        update = self.wait_for_update()
        if update.lines[0].source == 'First line':
            update = self.wait_for_update()
        self.assertEqual(update.lines[0].source, 'Third line')
        self.assertEqual(self.models.ocr_values, [1, 3])
        self.assertTrue(self.screen.wait_for_captures(3, count=4))
        self.assertEqual(self.models.ocr_values, [1, 3])
        self.assertIsNone(self.session.poll())

    def test_token_count_survives_cached_updates_only_within_the_same_run(self):
        self.models.last_token_count = 2255
        self.session.start(REGION)
        self.assertEqual(self.wait_for_update().token_count, 2255)
        self.screen.show(2)
        self.assertEqual(self.wait_for_update().token_count, 2255)
        self.session.stop()
        self.session.start(OTHER_REGION)
        self.assertIsNone(self.wait_for_update().token_count)
        self.assertEqual(self.models.batches, [['First line']])

    def test_region_change_discards_completion_from_previous_region(self):
        self.models.resume.clear()
        self.session.start(REGION)
        self.assertTrue(self.models.entered.wait(3))
        self.screen.show(3)
        self.session.start(OTHER_REGION)
        self.assertTrue(self.screen.wait_for_captures(3))
        self.models.resume.set()

        update = self.wait_for_update()
        self.assertEqual(update.region, OTHER_REGION)
        self.assertEqual(update.lines[0].source, 'Third line')
        self.assertIsNone(self.session.poll())

    def test_clear_discards_inflight_result_until_capture_changes(self):
        self.models.resume.clear()
        self.session.start(REGION)
        self.assertTrue(self.models.entered.wait(3))
        self.session.clear()
        self.assertEqual(self.wait_for_update().lines, ())
        self.models.resume.set()
        self.assertTrue(self.screen.wait_for_captures(1, count=4))
        self.assertIsNone(self.session.poll())

        self.screen.show(3)
        self.assertEqual(self.wait_for_update().lines[0].source, 'Third line')

    def test_stop_discards_inflight_result_and_restart_reanalyzes_same_pixels(self):
        self.models.resume.clear()
        self.session.start(REGION)
        self.assertTrue(self.models.entered.wait(3))
        self.session.stop()
        self.assertIsNone(self.session.poll())
        self.session.start(REGION)
        self.models.resume.set()
        self.assertEqual(self.wait_for_update().lines[0].source, 'First line')
        self.assertEqual(self.models.ocr_values, [1, 1])

    def test_capture_failure_is_reported_once_and_session_can_restart(self):
        self.screen.fail(RuntimeError('capture unavailable'))
        self.session.start(REGION)
        failure = self.wait_for_update()
        self.assertEqual(failure.region, REGION)
        self.assertEqual(failure.lines, ())
        self.assertIn('capture unavailable', failure.error)
        self.assertIsNone(self.session.poll())
        self.assertEqual(self.created, [])

        self.screen.fail(None)
        self.session.start(REGION)
        self.assertEqual(self.wait_for_update().lines[0].source, 'First line')

    def test_inference_failure_disposes_models_and_retry_creates_replacements(self):
        failing_models = self.models
        failing_models.failure = RuntimeError('model worker exited')
        self.session.start(REGION)
        self.assertIn('model worker exited', self.wait_for_update().error)
        self.assertIsNone(self.session.poll())
        self.assertTrue(failing_models.closed.wait(3))

        self.models = Models()
        self.session.start(REGION)
        self.assertEqual(self.wait_for_update().lines[0].source, 'First line')
        self.assertEqual(len(self.created), 2)

    def test_close_can_cancel_model_work_that_is_still_running(self):
        self.models.resume.clear()
        self.session.start(REGION)
        self.assertTrue(self.models.entered.wait(3))
        self.session.close()
        self.assertTrue(self.models.closed.wait(3))
        self.assertIsNone(self.session.poll())

    def test_capture_is_capped_at_configured_frame_rate(self):
        self.session.start(REGION)
        self.assertTrue(self.screen.wait_for_captures(1, count=4))
        first, fourth = self.screen.timestamps[0], self.screen.timestamps[3]
        self.assertGreaterEqual(fourth - first, 3 / 30 - 0.005)

    def test_empty_ocr_result_replaces_previously_translated_lines(self):
        self.session.start(REGION)
        self.assertEqual(len(self.wait_for_update().lines), 1)
        self.screen.show(0)
        self.assertEqual(self.wait_for_update().lines, ())

    def test_ocr_skips_language_neutral_lines_and_keeps_text_with_numbers(self):
        for value, text in enumerate(('8%', '$0.00', '?!', '8% off'), start=1):
            self.models.text[value] = text

        self.session.start(REGION)
        for value in (1, 2, 3):
            self.screen.show(value)
            self.assertEqual(self.wait_for_update().lines, ())

        self.screen.show(4)
        self.assertEqual(self.wait_for_update().lines[0].translation, 'VI 8% off')
        self.assertEqual(self.models.batches, [['8% off']])

    def test_ocr_skips_single_character_results_before_translation(self):
        sources = (
            'U', '盛', 'MA', 'C', 'O', 'の', 'S', 'y', 'C', 'B', 'Y', 'U',
            'M', 'U', '心', '人', 'L6', 'A', '8V', 'A', 'B', 'O',
            ' U. ', '「の」', 'はい', '人\nの心',
        )
        detected = [(BOX, source) for source in sources]
        lines, _ = self.session._translate(self.models, detected, OrderedDict(), preserve_line_breaks=True)

        expected = ['MA', 'L6', '8V', 'はい', '人\nの心']
        self.assertEqual(self.models.batches, [expected])
        self.assertEqual([line.source for line in lines], expected)


if __name__ == '__main__':
    unittest.main()



class DialogueFeed(Screen):
    def __init__(self):
        super().__init__()
        self.sources = {1: "How was your day?"}
        self.calls = []

    def ocr(self, frame):
        text = self.sources[int(frame[0, 0, 0])]
        return [(BOX, text)] if text else []

    def translate_contextual(self, text, *, role, rules, speaker, history):
        self.calls.append((text, tuple(source for _, source, _ in history)))
        return "VI " + text

    def close(self):
        pass

    def wait_for_update(self, session, text, region=REGION):
        deadline = time.monotonic() + 3
        expected = (text,) if text else ()
        while time.monotonic() < deadline:
            update = session.poll()
            if update is not None:
                assert update.error is None
                if update.region == region and tuple(line.source for line in update.lines) == expected:
                    return
            time.sleep(.005)
        raise AssertionError(f"Missing update for {text!r} at {region}")

    def show_text(self, session, text, region=REGION):
        value = len(self.sources) + 1
        self.sources[value] = text
        self.show(value)
        self.wait_for_update(session, text, region)


def test_moving_window_keeps_history_and_does_not_translate_the_same_frame():
    feed = DialogueFeed()
    session = LiveTranslationSession(lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2)
    try:
        session.start(REGION)
        feed.wait_for_update(session, "How was your day?")
        feed.show_text(session, "Second line")
        session.move(OTHER_REGION)
        feed.show_text(session, "Third line", OTHER_REGION)

        assert feed.calls[-1] == ("Third line", ("How was your day?", "Second line"))
        assert [text for text, _ in feed.calls].count("Second line") == 1
    finally:
        session.close()


def test_growing_dialogue_replaces_the_draft_in_history():
    feed = DialogueFeed()
    session = LiveTranslationSession(lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2)
    try:
        session.start(REGION)
        feed.wait_for_update(session, "How was your day?")
        for text in ("Hello", "Hello there", "Hello there!", "What next?"):
            feed.show_text(session, text)

        assert feed.calls[-2] == ("Hello there!", ("How was your day?",))
        assert feed.calls[-1] == ("What next?", ("How was your day?", "Hello there!"))
    finally:
        session.close()


def test_dialogue_with_shared_prefix_after_empty_frame_is_a_new_turn():
    feed = DialogueFeed()
    session = LiveTranslationSession(lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2)
    try:
        session.start(REGION)
        feed.wait_for_update(session, "How was your day?")
        for text in ("Wait", "", "Wait for me!", "What next?"):
            feed.show_text(session, text)

        assert feed.calls[-1] == ("What next?", ("Wait", "Wait for me!"))
    finally:
        session.close()


@pytest.mark.parametrize(("complete", "next_line"), [
    ("Wait.", "Wait. Let me explain."),
    ("「待って。」", "「待って。」説明させて。"),
])
def test_dialogue_with_shared_prefix_after_sentence_end_is_a_new_turn(complete, next_line):
    feed = DialogueFeed()
    session = LiveTranslationSession(lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2)
    try:
        session.start(REGION)
        feed.wait_for_update(session, "How was your day?")
        for text in (complete, next_line, "What next?"):
            feed.show_text(session, text)

        assert feed.calls[-1] == ("What next?", (complete, next_line))
    finally:
        session.close()


def test_move_publishes_in_flight_translation_at_new_position():
    translating, finish = threading.Event(), threading.Event()

    class BlockingFeed(DialogueFeed):
        def translate_contextual(self, text, **context):
            translating.set()
            assert finish.wait(3)
            return super().translate_contextual(text, **context)

    feed = BlockingFeed()
    session = LiveTranslationSession(lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2)
    try:
        session.start(REGION)
        assert translating.wait(3)
        session.move(OTHER_REGION)
        finish.set()
        feed.wait_for_update(session, "How was your day?", OTHER_REGION)
        assert feed.calls == [("How was your day?", ())]
    finally:
        finish.set()
        session.close()
