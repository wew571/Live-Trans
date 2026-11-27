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


@pytest.mark.parametrize("once", [False, True])
def test_ordered_crops_keep_speaker_and_recent_dialogue(once):
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

    def wait_translation(translation, request_id=None):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update:
                if once:
                    assert request_id is not None and update.request_id == request_id
                    assert [line.translation for line in update.lines] == ["Curren", translation]
                    return update.lines
                if any(line.translation == translation for line in update.lines):
                    return update.lines
            time.sleep(.01)
        raise AssertionError(f"Missing translation: {translation}")

    try:
        if not once:
            session.start(region)
        result = wait_translation("VI First line", session.request_once(region) if once else None)
        assert [line.box for line in result] == [(0, 0, 4, 4), (4, 0, 8, 4)]
        capture.dialogue = 3
        wait_translation("VI Next line", session.request_once(region) if once else None)
        assert [source for source, _ in models.calls] == ["カレン", "First line", "Next line"]
        assert models.calls[-1][1] == {
            "role": "dialogue", "rules": "Dialogue", "speaker": "Curren",
            "history": (("Curren", "First line", "VI First line"),),
        }
        session.clear_context()
        capture.dialogue = 4
        wait_translation("VI New scene", session.request_once(region) if once else None)
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

    def test_once_captures_before_loading_and_reuses_model_between_clicks(self):
        self.models.resume.clear()

        def factory():
            self.assertEqual(len(self.screen.timestamps), 1)
            self.screen.show(3)
            self.created.append(True)
            return self.models

        session = LiveTranslationSession(factory, capture_factory=self.screen.open)
        self.addCleanup(session.close)
        self.addCleanup(self.models.resume.set)
        first_id = session.request_once(REGION)
        self.assertIsNotNone(first_id)
        self.assertTrue(self.models.entered.wait(3))
        self.assertIsNone(session.request_once(REGION))
        self.assertFalse(self.screen.wait_for_captures(3, 1))
        self.models.resume.set()

        self.session = session
        first = self.wait_for_update()
        self.assertEqual((first.request_id, first.lines[0].translation), (first_id, 'VI First line'))
        second_id = session.request_once(REGION)
        second = self.wait_for_update()
        self.assertNotEqual(first_id, second_id)
        self.assertEqual((second.request_id, second.lines[0].translation), (second_id, 'VI Third line'))
        third_id = session.request_once(REGION)
        third = self.wait_for_update()
        self.assertEqual((third.request_id, third.lines), (third_id, second.lines))
        self.assertEqual(len(self.screen.timestamps), 3)
        self.assertEqual(self.created, [True])
        self.assertEqual(self.models.batches, [['First line'], ['Third line']])

    def test_once_times_out_waiting_for_window_frame_and_can_retry(self):
        attempts = []
        screen = self.screen

        class Capture:
            def grab(self, region):
                attempts.append(region)
                if ready and len(attempts) > first_attempt + 1:
                    return screen.grab(region)
                return None

        @contextmanager
        def open_capture():
            yield Capture()

        ready = False
        self.session = LiveTranslationSession(
            lambda: self.models, capture_factory=open_capture, once_timeout=.1,
        )
        self.addCleanup(self.session.close)
        failed_id = self.session.request_once(REGION)
        failed = self.wait_for_update()
        self.assertEqual(failed.request_id, failed_id)
        self.assertIn('Timed out', failed.error)
        self.assertEqual(self.models.ocr_values, [])

        first_attempt = len(attempts)
        ready = True
        retry_id = self.session.request_once(REGION)
        retry = self.wait_for_update()
        self.assertEqual(retry.request_id, retry_id)
        self.assertIsNone(retry.error)
        self.assertEqual(len(attempts), first_attempt + 2)

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

    def test_close_cancels_inflight_once_request(self):
        self.models.resume.clear()
        self.assertIsNotNone(self.session.request_once(REGION))
        self.assertTrue(self.models.entered.wait(3))
        self.session.close()
        self.assertTrue(self.models.closed.wait(3))
        self.models.resume.set()
        self.assertIsNone(self.session.request_once(REGION))
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
                    return update
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

def test_live_delay_starts_after_render_ack_and_keeps_latest_frame():
    screen, models = Screen(), Models()
    session = LiveTranslationSession(lambda: models, capture_factory=screen.open, post_translation_delay_ms=150)
    try:
        session.start(REGION)
        deadline = time.monotonic() + 3
        update = None
        while update is None and time.monotonic() < deadline:
            update = session.poll()
            time.sleep(.005)
        assert update and update.render_id is not None
        screen.show(2)
        assert screen.wait_for_captures(2)
        time.sleep(.2)
        assert models.ocr_values == [1]
        session.acknowledge_render(update.render_id)
        screen.show(3)
        assert screen.wait_for_captures(3)
        time.sleep(.05)
        assert models.ocr_values == [1]
        deadline = time.monotonic() + 3
        while len(models.ocr_values) < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert models.ocr_values == [1, 3]
    finally:
        session.close()


def test_configure_live_cancels_gate_and_reuses_models():
    screen, models = Screen(), Models()
    created = []
    def factory():
        created.append(True)
        return models
    session = LiveTranslationSession(factory, capture_factory=screen.open, post_translation_delay_ms=1000)
    try:
        session.start(REGION)
        deadline = time.monotonic() + 3
        update = None
        while update is None and time.monotonic() < deadline:
            update = session.poll()
            time.sleep(.005)
        assert update and update.render_id is not None
        screen.show(3)
        assert screen.wait_for_captures(3)
        session.configure_live(frame_crop=None, frame_crops=(), history_turns=0, post_translation_delay_ms=0)
        deadline = time.monotonic() + 3
        while len(models.ocr_values) < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(models.ocr_values) >= 2
        assert created == [True]
    finally:
        session.close()

def test_multicrop_delay_arms_only_after_final_crop():
    from core.live_translation import WindowCrop
    screen, models = Screen(), Models()
    session = LiveTranslationSession(
        lambda: models, capture_factory=screen.open, post_translation_delay_ms=100,
        frame_crops=((WindowCrop(0, 0, .5, 1), 'dialogue', ''),
                     (WindowCrop(.5, 0, 1, 1), 'dialogue', '')),
    )
    try:
        session.start(REGION)
        deadline = time.monotonic() + 3
        update = None
        while update is None and time.monotonic() < deadline:
            update = session.poll()
            time.sleep(.005)
        assert update and len(update.lines) == 2 and update.render_id is not None
        assert models.ocr_values == [1, 1]
        screen.show(3)
        assert screen.wait_for_captures(3)
        time.sleep(.15)
        assert models.ocr_values == [1, 1]
        session.acknowledge_render(update.render_id)
        deadline = time.monotonic() + 3
        while len(models.ocr_values) < 4 and time.monotonic() < deadline:
            time.sleep(.005)
        assert models.ocr_values == [1, 1, 3, 3]
    finally:
        session.close()

def test_configure_history_truncates_and_crop_change_clears_context():
    from core.live_translation import WindowCrop
    screen = Screen()
    class ContextModels(Models):
        def __init__(self):
            super().__init__()
            self.contexts = []
        def translate_contextual(self, text, **context):
            self.contexts.append(context['history'])
            return 'VI ' + text
    models = ContextModels()
    session = LiveTranslationSession(lambda: models, capture_factory=screen.open, group_dialogue=True, history_turns=2)
    def once(value):
        screen.show(value)
        request = session.request_once(REGION)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update and update.request_id == request:
                return
            time.sleep(.005)
        raise AssertionError('one-shot did not complete')
    try:
        models.text.update({2: 'Second line', 4: 'Fourth line'})
        once(1)
        once(2)
        session.configure_live(frame_crop=None, frame_crops=(), history_turns=1, post_translation_delay_ms=0)
        once(3)
        assert models.contexts[-1] == (('', 'Second line', 'VI Second line'),)
        session.configure_live(frame_crop=WindowCrop(0, 0, .5, 1), frame_crops=(), history_turns=1, post_translation_delay_ms=0)
        once(4)
        assert models.contexts[-1] == ()
    finally:
        session.close()

def test_hot_history_change_and_context_preserving_resume_do_not_duplicate_dialogue():
    screen = Screen()
    class ContextModels(Models):
        def __init__(self):
            super().__init__()
            self.contexts = []
        def translate_contextual(self, text, **context):
            self.contexts.append((text, context['history']))
            return 'VI ' + text
    models = ContextModels()
    models.text.update({2: 'Second line', 3: 'Third line'})
    session = LiveTranslationSession(lambda: models, capture_factory=screen.open, group_dialogue=True, history_turns=2)
    def wait_update(text):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update and any(line.source == text for line in update.lines):
                return update
            time.sleep(.005)
        raise AssertionError('missing ' + text)
    try:
        session.start(REGION)
        wait_update('First line')
        screen.show(2)
        wait_update('Second line')
        session.configure_live(frame_crop=None, frame_crops=(), history_turns=1, post_translation_delay_ms=0)
        wait_update('Second line')
        session.stop(reset_context=False)
        session.start(REGION, reset_context=False)
        wait_update('Second line')
        screen.show(3)
        wait_update('Third line')
        assert models.contexts == [
            ('First line', ()),
            ('Second line', (('', 'First line', 'VI First line'),)),
            ('Third line', (('', 'Second line', 'VI Second line'),)),
        ]
    finally:
        session.close()


def test_duplicate_render_ack_does_not_extend_delay():
    screen, models = Screen(), Models()
    session = LiveTranslationSession(lambda: models, capture_factory=screen.open, post_translation_delay_ms=300)
    try:
        session.start(REGION)
        deadline = time.monotonic() + 3
        update = None
        while update is None and time.monotonic() < deadline:
            update = session.poll()
            time.sleep(.005)
        assert update and update.render_id is not None
        screen.show(3)
        assert screen.wait_for_captures(3)
        session.acknowledge_render(update.render_id)
        time.sleep(.22)
        session.acknowledge_render(update.render_id)
        deadline = time.monotonic() + .2
        while len(models.ocr_values) < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert models.ocr_values == [1, 3]
    finally:
        session.close()

@pytest.mark.parametrize('history,delay', [(0, 100), (2, 0)])
def test_configure_preserves_completed_unpolled_once(history, delay):
    screen, models = Screen(), Models()
    session = LiveTranslationSession(lambda: models, capture_factory=screen.open)
    try:
        request = session.request_once(REGION)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with session._state:
                ready = session._update is not None
            if ready:
                break
            time.sleep(.005)
        assert ready
        session.configure_live(frame_crop=None, frame_crops=(), history_turns=history, post_translation_delay_ms=delay)
        update = session.poll()
        assert update is not None and update.request_id == request
    finally:
        session.close()

@pytest.mark.parametrize(("initial_history", "saved_history"), [(2, 1), (0, 1), (2, 0)])
def test_saving_history_reprocesses_the_original_one_shot_frame(initial_history, saved_history):
    class SplitScreen(Screen):
        def grab(self, region):
            frame = super().grab(region)
            frame[:, :2] = 9
            return frame

    class ContextModels(Models):
        def translate(self, texts):
            return ['VI without history: ' + text for text in texts]

        def translate_contextual(self, text, **context):
            return 'VI with history: ' + text

    screen, models = SplitScreen(), ContextModels()
    crop = live_translation.WindowCrop(.5, 0, 1, 1)
    session = LiveTranslationSession(
        lambda: models, capture_factory=screen.open, frame_crop=crop,
        group_dialogue=True, history_turns=initial_history,
    )
    models.resume.clear()
    try:
        request = session.request_once(REGION)
        assert request is not None and models.entered.wait(3)
        screen.show(3)
        session.configure_live(
            frame_crop=crop, frame_crops=(), history_turns=saved_history, post_translation_delay_ms=0,
        )
        models.resume.set()

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update is not None:
                assert update.error is None and update.request_id == request
                assert [line.source for line in update.lines] == ['First line']
                mode = 'with history' if saved_history else 'without history'
                assert [line.translation for line in update.lines] == [f'VI {mode}: First line']
                assert models.ocr_values == [1, 1]
                assert len(screen.timestamps) == 1
                return
            time.sleep(.005)
        raise AssertionError('one-shot did not complete')
    finally:
        models.resume.set()
        session.close()


def test_saving_history_does_not_discard_a_one_shot_capture_in_progress():
    captured, release = threading.Event(), threading.Event()

    class BlockingScreen(Screen):
        def grab(self, region):
            frame = super().grab(region)
            captured.set()
            assert release.wait(3)
            return frame

    screen, models = BlockingScreen(), Models()
    session = LiveTranslationSession(
        lambda: models, capture_factory=screen.open, history_turns=2,
    )
    try:
        request = session.request_once(REGION)
        assert request is not None and captured.wait(3)
        screen.show(3)
        session.configure_live(
            frame_crop=None, frame_crops=(), history_turns=1, post_translation_delay_ms=0,
        )
        release.set()

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update is not None:
                assert update.error is None and update.request_id == request
                assert [line.source for line in update.lines] == ['First line']
                assert models.ocr_values == [1]
                assert len(screen.timestamps) == 1
                return
            time.sleep(.005)
        raise AssertionError('one-shot did not complete')
    finally:
        release.set()
        session.close()


def test_clear_stays_hidden_after_hot_configuration_on_unchanged_frame():
    screen = Screen()
    class ContextModels(Models):
        def translate_contextual(self, text, **context):
            return 'VI ' + text
    models = ContextModels()
    session = LiveTranslationSession(lambda: models, capture_factory=screen.open, group_dialogue=True, history_turns=2)
    try:
        session.start(REGION)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and session.poll() is None:
            time.sleep(.005)
        session.clear()
        session.clear_context()
        assert session.poll().lines == ()
        session.configure_live(frame_crop=None, frame_crops=(), history_turns=1, post_translation_delay_ms=0)
        assert screen.wait_for_captures(1, 3)
        time.sleep(.15)
        assert session.poll() is None
        screen.show(3)
        deadline = time.monotonic() + 3
        update = None
        while time.monotonic() < deadline:
            update = session.poll()
            if update and update.lines:
                break
            time.sleep(.005)
        assert update and update.lines[0].source == 'Third line'
    finally:
        session.close()

def test_clear_remains_hidden_when_only_outside_crops_changes():
    from core.live_translation import WindowCrop
    from contextlib import nullcontext
    class Capture:
        outside = 0
        dialogue = 1
        def grab(self, region):
            frame = np.full((3, 4, 3), self.outside, dtype=np.uint8)
            frame[:, :2] = self.dialogue
            return frame
    capture, models = Capture(), Models()
    session = LiveTranslationSession(
        lambda: models, capture_factory=lambda: nullcontext(capture), group_dialogue=True,
        frame_crops=((WindowCrop(0, 0, .5, 1), 'dialogue', ''),), history_turns=2,
    )
    models.translate_contextual = lambda text, **context: 'VI ' + text
    try:
        session.start(REGION)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and session.poll() is None:
            time.sleep(.005)
        session.clear()
        assert session.poll().lines == ()
        capture.outside = 2
        time.sleep(.2)
        assert session.poll() is None
        capture.dialogue = 3
        deadline = time.monotonic() + 3
        update = None
        while time.monotonic() < deadline:
            update = session.poll()
            if update and update.lines:
                break
            time.sleep(.005)
        assert update and update.lines[0].source == 'Third line'
    finally:
        session.close()


@pytest.mark.parametrize(
    ("once", "delay", "newer_frame"),
    [(False, 0, False), (False, 20, False), (False, 20, True), (True, 20, False)],
)
def test_clear_context_retries_in_flight_crops_with_empty_history(once, delay, newer_frame):
    entered, release = threading.Event(), threading.Event()

    class ContextModels(Models):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.text[2] = "Second line"

        def translate_contextual(self, text, *, role, rules, speaker, history):
            self.calls.append((text, rules, tuple(source for _, source, _ in history)))
            if text == "Second line" and rules == "left" and not entered.is_set():
                entered.set()
                assert release.wait(3)
            return f"{len(history)} turns: {text}"

    screen, models = Screen(), ContextModels()
    crops = (
        (live_translation.WindowCrop(0, 0, .5, 1), "dialogue", "left"),
        (live_translation.WindowCrop(.5, 0, 1, 1), "dialogue", "right"),
    )
    session = LiveTranslationSession(
        lambda: models, capture_factory=screen.open, group_dialogue=True,
        frame_crops=crops, history_turns=4, post_translation_delay_ms=delay,
    )

    def wait_update(source, expected_translations):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update is not None:
                assert update.error is None
                for index, line in enumerate(update.lines):
                    if line.source == source:
                        assert line.translation == expected_translations[index]
                if tuple(line.source for line in update.lines) == (source, source):
                    return update
            time.sleep(.005)
        raise AssertionError("Translation did not finish")

    try:
        if once:
            first_request = session.request_once(REGION)
        else:
            first_request = None
            session.start(REGION)
        baseline = wait_update("First line", ["0 turns: First line", "1 turns: First line"])
        assert baseline.request_id == first_request
        if baseline.render_id is not None:
            session.acknowledge_render(baseline.render_id)

        screen.show(2)
        request = session.request_once(REGION) if once else None
        assert entered.wait(3)
        assert models.calls[-1] == ("Second line", "left", ("First line", "First line"))
        if once:
            screen.show(3)
        else:
            session.move(OTHER_REGION)
            if newer_frame:
                screen.show(3)
                assert screen.wait_for_captures(3)
        session.clear_context()
        assert session.poll() is None
        release.set()

        source = "Third line" if newer_frame else "Second line"
        update = wait_update(source, [f"0 turns: {source}", f"1 turns: {source}"])
        assert update.request_id == request
        assert update.region == (REGION if once else OTHER_REGION)
        assert models.calls[3:] == [
            (source, "left", ()),
            (source, "right", (source,)),
        ]
        assert (update.render_id is not None) == (not once and delay > 0)
    finally:
        release.set()
        session.close()


def test_clear_context_drops_an_unpolled_live_update_and_its_render_gate():
    feed = DialogueFeed()
    session = LiveTranslationSession(
        lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2,
        post_translation_delay_ms=20,
    )
    try:
        session.start(REGION)
        baseline = feed.wait_for_update(session, "How was your day?")
        session.acknowledge_render(baseline.render_id)
        feed.sources[2] = "Next line"
        feed.show(2)
        deadline = time.monotonic() + 3
        stale_render_id = None
        while stale_render_id is None and time.monotonic() < deadline:
            with session._state:
                if session._update is not None:
                    stale_render_id = session._update.render_id
            time.sleep(.005)
        assert stale_render_id is not None
        session.clear_context()

        refreshed = feed.wait_for_update(session, "Next line")
        assert feed.calls[-1] == ("Next line", ())
        assert refreshed.render_id is not None and refreshed.render_id != stale_render_id
    finally:
        session.close()


def test_idle_context_clear_does_not_put_unchanged_dialogue_back_in_history():
    class TwoCropFeed(DialogueFeed):
        def grab(self, region):
            frame = super().grab(region)
            frame[:, :2] = 1
            return frame

    feed = TwoCropFeed()
    feed.sources.update({2: "Current topic", 3: "New topic"})
    session = LiveTranslationSession(
        lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=4,
        frame_crops=(
            (live_translation.WindowCrop(0, 0, .5, 1), "dialogue", ""),
            (live_translation.WindowCrop(.5, 0, 1, 1), "dialogue", ""),
        ),
        post_translation_delay_ms=20,
    )

    def wait_right(source):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            update = session.poll()
            if update is not None:
                assert tuple(line.source for line in update.lines) == ("How was your day?", source)
                return update
            time.sleep(.005)
        raise AssertionError(f"Missing translation for {source!r}")

    try:
        feed.show(2)
        session.start(REGION)
        baseline = wait_right("Current topic")
        session.acknowledge_render(baseline.render_id)
        session.clear_context()
        assert session.poll() is None
        feed.show(3)
        wait_right("New topic")
        assert feed.calls == [
            ("How was your day?", ()),
            ("Current topic", ("How was your day?",)),
            ("New topic", ()),
        ]
    finally:
        session.close()


def test_context_clear_keeps_a_completed_unpolled_one_shot():
    feed = DialogueFeed()
    session = LiveTranslationSession(
        lambda: feed, capture_factory=feed.open, group_dialogue=True, history_turns=2,
    )
    try:
        request = session.request_once(REGION)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with session._state:
                if session._update is not None:
                    break
            time.sleep(.005)
        session.clear_context()
        completed = session.poll()
        assert completed is not None and completed.request_id == request
        assert [line.source for line in completed.lines] == ["How was your day?"]
    finally:
        session.close()
