from contextlib import nullcontext
import json
import subprocess
import sys
import textwrap
from threading import Thread
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def client_type():
    from core.live_models import LiveModelClient

    return LiveModelClient


@pytest.fixture
def child_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("MANGALENS_LIVE_PYTHON", sys.executable)
    monkeypatch.setenv("MANGALENS_LIVE_PYTHONPATH", str(tmp_path))

    def write_module(name, source):
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")

    return write_module


def test_invalid_or_empty_input_does_not_launch_worker(monkeypatch, client_type):
    monkeypatch.setenv("MANGALENS_LIVE_PYTHON", "missing-inference-executable")
    client = client_type()
    try:
        assert client.translate([]) == []
        with pytest.raises(ValueError, match="BGR"):
            client.ocr(np.zeros((4, 4, 4), dtype=np.uint8))
        with pytest.raises(ValueError, match="BGR"):
            client.ocr(np.zeros((4, 4, 3), dtype=np.float32))
        with pytest.raises(TypeError, match="strings"):
            client.translate([42])
    finally:
        client.close()


def test_live_worker_skips_provider_probes_when_loading_local_weights(monkeypatch):
    from core.live_models import _worker_environment

    flag = "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"
    monkeypatch.delenv(flag, raising=False)
    assert _worker_environment()[flag] == "True"


@pytest.fixture
def worker_translator(child_modules, tmp_path, monkeypatch):
    from core import live_models

    child_modules("test_worker", """
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path.cwd() / "core"))
        import live_inference_worker
        from test_translator import build

        live_inference_worker.load_translator = build
        live_inference_worker.main()
    """)
    monkeypatch.setattr(live_models, "_WORKER", tmp_path / "test_worker.py")
    return lambda source: child_modules("test_translator", source)


def test_worker_transports_unicode_batches_even_if_inference_prints(client_type, worker_translator):
    worker_translator(r"""
        import os

        def build():
            print('loading a custom model', flush=True)
            def translate(texts):
                os.write(1, b"native inference message\n")
                return [f'Bản dịch: {text}' for text in texts]
            return translate
    """)
    client = client_type()
    try:
        assert client.translate(["café\n🧪", "second"]) == ["Bản dịch: café\n🧪", "Bản dịch: second"]
        assert client.translate(["again"]) == ["Bản dịch: again"]
    finally:
        client.close()


def test_live_routes_japanese_lines_and_preserves_mixed_batch_order(monkeypatch):
    from core import live_translator

    loads = []

    def load_japanese():
        loads.append("Mitsua")

        def translate_japanese(texts):
            assert texts == ["設定", "ゲームを開始", "Press E: 扉を開ける"]
            return ["Settings", "Start game", "Press E to open the door"]

        return translate_japanese

    def load_english():
        loads.append("VinAI")

        def translate_english(texts):
            vietnamese = {
                "Steam Library": "Thư viện Steam", "Continue": "Tiếp tục", "Settings": "Cài đặt", "Start game": "Bắt đầu trò chơi",
                "Exit": "Thoát", "Press E to open the door": "Nhấn E để mở cửa",
            }
            return [vietnamese[text] for text in texts]

        return translate_english

    monkeypatch.delenv("MANGALENS_LIVE_TRANSLATOR_FACTORY", raising=False)
    monkeypatch.setattr(live_translator, "_load_mbart_translator", load_english)
    monkeypatch.setattr(live_translator, "_load_mitsua_translator", load_japanese, raising=False)
    monkeypatch.setattr(live_translator, "_load_opus_translator", lambda: pytest.fail("OPUS-MT should not load"), raising=False)
    translate = live_translator.load_translator()

    assert not loads
    assert translate([]) == []
    assert not loads
    assert translate(["Steam Library", "Exit"]) == ["Thư viện Steam", "Thoát"]
    assert loads == ["VinAI"]
    assert translate(["設定", "ｹﾞｰﾑを開始", "Press E: 扉を開ける"]) == [
        "Cài đặt", "Bắt đầu trò chơi", "Nhấn E để mở cửa"
    ]
    assert loads == ["VinAI", "Mitsua"]
    source = ["Continue", "設定", "ｹﾞｰﾑを開始", "Exit", "Press E: 扉を開ける", "한글"]
    expected = ["Tiếp tục", "Cài đặt", "Bắt đầu trò chơi", "Thoát", "Nhấn E để mở cửa", ""]
    assert translate(source) == expected
    assert translate(source) == expected
    assert loads == ["VinAI", "Mitsua"]


def test_worker_uses_fixed_model_weights_and_tokenizers(monkeypatch, client_type, child_modules):
    child_modules("torch", """
        from contextlib import nullcontext

        float32 = 'torch.float32'
        inference_mode = nullcontext

        def set_num_threads(threads):
            assert threads == 4

        class cuda:
            @staticmethod
            def is_available():
                return False
    """)
    child_modules("transformers", """
        class Inputs(dict):
            def to(self, device):
                assert device == 'cpu'
                return self

        class AutoTokenizer:
            unk_token_id = 0

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                if 'Mitsua' in name:
                    assert 'elan-mt-bt-ja-en' in name and 'src_lang' not in kwargs
                else:
                    assert 'vinai-translate-en2vi-v2' in name
                    assert kwargs['src_lang'] == 'en_XX'
                return cls()

            def convert_tokens_to_ids(self, language):
                assert language == 'vi_VN'
                return 17

            def __call__(self, texts, **kwargs):
                assert kwargs['padding'] is True
                return Inputs(texts=texts)

            def batch_decode(self, output, skip_special_tokens):
                assert skip_special_tokens is True
                return output

        class AutoModelForSeq2SeqLM:
            class config:
                max_position_embeddings = 1024

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                assert 'elan-mt-bt-ja-en' in name or 'vinai-translate-en2vi-v2' in name
                assert kwargs['dtype'] == 'torch.float32'
                return cls()

            def eval(self):
                return self

            def to(self, device):
                assert device == 'cpu'
                return self

            def generate(self, *, texts, num_beams, do_sample, decoder_start_token_id=None, **kwargs):
                assert num_beams == 1 and do_sample is False
                if decoder_start_token_id is None:
                    assert texts == ['いち', 'に']
                    return ['one', 'two']
                assert decoder_start_token_id == 17
                return ['dịch ' + text for text in texts]
    """)
    monkeypatch.setenv("MANGALENS_LIVE_TRANSLATOR_FACTORY", "missing:factory")
    monkeypatch.setenv("JA_ELANMT_MODEL", "example/japanese-english")
    monkeypatch.setenv("MANGALENS_LIVE_TRANSLATOR_MODEL", "example/english-vietnamese")
    monkeypatch.setenv("en_vi_weight", "manga/original-weights")
    monkeypatch.setenv("en_vi_token", "manga/original-tokenizer")
    client = client_type()
    try:
        assert client.translate(["いち", "Ignore", "に"]) == ["dịch one", "dịch Ignore", "dịch two"]
    finally:
        client.close()


@pytest.mark.parametrize(
    ("bundle", "use_cuda", "accelerate", "use_local_weights"),
    [
        ("complete", True, True, True),
        ("complete", False, True, False),
        ("complete", True, False, False),
        ("unset", True, True, False),
        ("missing-index", True, True, False),
        ("bad-index", True, True, False),
        ("missing-manifest", True, True, False),
        ("wrong-model", True, True, False),
        ("wrong-tokenizer", True, True, False),
        ("missing-shard", True, True, False),
        ("truncated-shard", True, True, False),
        ("wrong-index-tensor", True, True, False),
    ],
)
def test_live_mbart_selects_complete_fp16_checkpoint_on_cuda(
    monkeypatch, tmp_path, bundle, use_cuda, accelerate, use_local_weights
):
    from core import live_translator

    checkpoint = tmp_path / "models" / "live-mbart-fp16"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    if bundle != "missing-manifest":
        (checkpoint / "mangalens-source.json").write_text(json.dumps({
            "model": "example/other-model" if bundle == "wrong-model" else "vinai/vinai-translate-en2vi-v2",
            "tokenizer": "example/other-tokenizer" if bundle == "wrong-tokenizer" else "vinai/vinai-translate-en2vi-v2",
        }), encoding="utf-8")
    index = checkpoint / "model.safetensors.index.json"
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    if bundle == "bad-index":
        index.write_text("{}", encoding="utf-8")
    elif bundle != "missing-index":
        second_weight = "unavailable" if bundle == "wrong-index-tensor" else "weight2"
        index.write_text(
            json.dumps({"weight_map": {"weight1": shards[0], second_weight: shards[1]}}),
            encoding="utf-8",
        )
    for position, shard in enumerate(shards):
        if position == 1 and bundle == "missing-shard":
            continue
        header = json.dumps({
            f"weight{position + 1}": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        }).encode("utf-8")
        data = b"" if position == 1 and bundle == "truncated-shard" else b"\x00\x00"
        (checkpoint / shard).write_bytes(len(header).to_bytes(8, "little") + header + data)

    monkeypatch.setattr(live_translator, "_ROOT", tmp_path)
    monkeypatch.setattr(
        live_translator, "resolve_model", lambda *_: "vinai/vinai-translate-en2vi-v2"
    )
    if bundle == "unset":
        checkpoint.rename(checkpoint.with_name("unconfigured"))

    device = "cuda:0" if use_cuda else "cpu"
    loads = []
    moves = []

    class Inputs(dict):
        def to(self, target):
            assert target == device
            return self

    class Tokenizer:
        unk_token_id = 0

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            assert source == "vinai/vinai-translate-en2vi-v2"
            assert kwargs == {"local_files_only": True, "src_lang": "en_XX"}
            return cls()

        def convert_tokens_to_ids(self, language):
            assert language == "vi_VN"
            return 17

        def __call__(self, texts, **kwargs):
            return Inputs(texts=texts)

        def batch_decode(self, outputs, *, skip_special_tokens):
            return outputs

    class Model:
        config = SimpleNamespace(max_position_embeddings=1024)

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            loads.append((source, kwargs))
            return cls()

        def eval(self):
            return self

        def to(self, target):
            moves.append(target)
            return self

        def generate(self, *, texts, decoder_start_token_id, **kwargs):
            assert decoder_start_token_id == 17
            return ["dịch " + text for text in texts]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: use_cuda),
        float16="float16", float32="float32",
        inference_mode=nullcontext, set_num_threads=lambda _: None,
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Tokenizer, AutoModelForSeq2SeqLM=Model,
    ))
    monkeypatch.setitem(sys.modules, "transformers.utils", SimpleNamespace(
        is_accelerate_available=lambda: accelerate,
    ))

    assert live_translator._load_mbart_translator()(["one"]) == ["dịch one"]
    assert len(loads) == 1
    source, options = loads[0]
    assert options["local_files_only"] is True
    assert options["dtype"] == ("float16" if use_cuda else "float32")
    if use_local_weights:
        assert str(source) == str(checkpoint)
        assert options["device_map"] == "cuda:0"
        assert not moves
    else:
        assert source == "vinai/vinai-translate-en2vi-v2"
        assert "device_map" not in options
        assert moves == [device]


def test_ocr_sends_noncontiguous_bgr_and_clips_local_boxes(client_type, child_modules):
    child_modules("cv2", """
        def setNumThreads(threads):
            assert threads == 4
    """)
    child_modules("torch", """
        def set_num_threads(threads):
            assert threads == 4

        class cuda:
            @staticmethod
            def is_available():
                return False
    """)
    child_modules("paddleocr", """
        class PaddleOCR:
            def __init__(self, **kwargs):
                assert kwargs['device'] == 'cpu'
                assert kwargs['engine'] == 'transformers'
                assert kwargs['engine_config']['dtype'] == 'float32'
                assert kwargs['text_detection_model_name'] == 'PP-OCRv6_small_det'
                assert kwargs['text_recognition_model_name'] == 'PP-OCRv6_small_rec'

            def predict(self, frame, **kwargs):
                assert kwargs == {'text_det_limit_side_len': 960, 'text_det_limit_type': 'max'}
                assert frame.shape == (4, 5, 3)
                assert frame[1, 1].tolist() == [19, 23, 29]
                return [{
                    'rec_texts': ['  hello ', '', 'outside'],
                    'rec_polys': [
                        [[-2, 2], [3, -1], [7, 2], [1, 7]],
                        [[1, 1], [3, 1], [3, 3], [1, 3]],
                        [[8, 8], [9, 8], [9, 9], [8, 9]],
                    ],
                }]
    """)
    source = np.zeros((4, 10, 3), dtype=np.uint8)
    source[1, 2] = (19, 23, 29)
    client = client_type()
    try:
        assert client.ocr(source[:, ::2]) == [((0, 0, 5, 4), "hello")]
    finally:
        client.close()


def test_failed_translation_has_actionable_error_and_reaps_worker(
    monkeypatch, client_type, worker_translator
):
    worker_translator("""
        def build():
            return lambda texts: []
    """)
    processes = []
    popen = subprocess.Popen

    def record_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", record_process)
    client = client_type()
    try:
        with pytest.raises(RuntimeError, match="one string per input"):
            client.translate(["first"])
        assert len(processes) == 1
        assert processes[0].poll() is not None
        with pytest.raises(RuntimeError, match="closed"):
            client.translate(["second"])
    finally:
        client.close()
        client.close()


def test_worker_timeout_reaps_running_child(client_type, worker_translator):
    worker_translator("""
        import time

        def build():
            def translate(texts):
                time.sleep(10)
                return texts
            return translate
    """)
    client = client_type(request_timeout=0.5)
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            client.translate(["first"])
        with pytest.raises(RuntimeError, match="closed"):
            client.translate(["second"])
    finally:
        client.close()


def test_close_interrupts_started_inference(monkeypatch, client_type, worker_translator, tmp_path):
    ready = tmp_path / "started"
    worker_translator("""
        import os
        from pathlib import Path
        from threading import Event

        def build():
            def translate(texts):
                Path(os.environ['MODEL_TEST_READY']).touch()
                Event().wait()
            return translate
    """)
    monkeypatch.setenv("MODEL_TEST_READY", str(ready))
    errors = []
    client = client_type()

    def translate():
        try:
            client.translate(["first"])
        except RuntimeError as error:
            errors.append(error)

    inference = Thread(target=translate, daemon=True)
    closer = Thread(target=client.close, daemon=True)
    inference.start()
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), errors
    finally:
        closer.start()
        closer.join(5)
        inference.join(5)

    assert not closer.is_alive()
    assert not inference.is_alive()
    assert len(errors) == 1
    with pytest.raises(RuntimeError, match="closed"):
        client.translate([])
