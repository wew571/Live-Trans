from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    for key in (
        "YOLO_weight", "HF_HOME", "MANGA_LANGUAGE_DETECTION_MODEL", "ZH_READ_MODEL",
        "JA_ELANMT_MODEL", "ZH_TRANS_MODEL", "KR_TRANS_MODEL", "en_vi_weight", "en_vi_token",
        "MANGALENS_LIVE_TRANSLATOR_MODEL", "MANGALENS_LIVE_TRANSLATOR_FACTORY",
        "MANGALENS_LIVE_DET_DIR", "MANGALENS_LIVE_REC_DIR", "HF_HOME", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "PADDLE_PDX_CACHE_HOME", "HF_TOKEN", "HF_ENDPOINT",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


def _asset(path, content=b"model data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _items(root):
    from core.setup_models import model_items

    return {item.key: item for item in model_items(root)}


def test_default_inventory_and_scan_stay_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    items = _items(tmp_path)
    assert items.keys() == {
        "translate-ja-en", "live-translate-en-vi", "live-ocr-det", "live-ocr-rec",
    }
    assert not any(item.is_ready() for item in items.values())
    assert list(tmp_path.iterdir()) == []


def test_incomplete_cache_cannot_hide_full_existing_snapshot(tmp_path):
    from core.setup_models import resolve_model

    repository = tmp_path / "model" / "hub" / "models--Mitsua--elan-mt-bt-ja-en"
    snapshot = repository / "snapshots" / ("a" * 40)
    current = repository / "refs" / "main"
    _asset(current, b"b" * 40)
    _asset(repository / "snapshots" / ("b" * 40) / "config.json", b"{}")
    for file in ("config.json", "pytorch_model.bin", "tokenizer_config.json",
                 "source.spm", "target.spm", "vocab.json"):
        _asset(snapshot / file, b"{}" if file.endswith(".json") else b"model data")
    assert _items(tmp_path)["translate-ja-en"].is_ready()
    assert resolve_model(tmp_path, "translate-ja-en") == str(snapshot)


def test_live_model_dir_needs_all_four_transformers_files(tmp_path):
    override = tmp_path / "models" / "live-ocr" / "PP-OCRv6_small_rec_safetensors"
    _asset(override / "config.json", b"{}")
    _asset(override / "preprocessor_config.json", b"{}")
    _asset(override / "inference.yml")
    _asset(override / "model.safetensors", b"version https://git-lfs.github.com/spec/v1\n")
    assert not _items(tmp_path)["live-ocr-rec"].is_ready()


def test_old_model_settings_do_not_change_fixed_sources(tmp_path, monkeypatch):
    from core.setup_models import resolve_model

    (tmp_path / ".env").write_text(
        "JA_ELANMT_MODEL=old/japanese\nMANGALENS_LIVE_TRANSLATOR_MODEL=old/english\n"
        "en_vi_weight=old/en-vi\nen_vi_token=old/tokenizer\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MANGALENS_LIVE_TRANSLATOR_FACTORY", "custom:load")
    monkeypatch.setenv("MANGALENS_LIVE_REC_DIR", str(tmp_path / "old-ocr"))

    assert resolve_model(tmp_path, "translate-ja-en") == "Mitsua/elan-mt-bt-ja-en"
    assert resolve_model(tmp_path, "live-translate-en-vi") == "vinai/vinai-translate-en2vi-v2"
    assert resolve_model(tmp_path, "live-ocr-rec") == str(
        tmp_path / "models" / "live-ocr" / "PP-OCRv6_small_rec_safetensors"
    )
    assert set(_items(tmp_path)) == {
        "translate-ja-en", "live-translate-en-vi", "live-ocr-det", "live-ocr-rec",
    }


_WEIGHT_DATA = b"realistic-model-data-for-http-resume"
_REVISION = "a" * 40
_MITSUA_FILES = {
    "config.json": b"{}",
    "pytorch_model.bin": _WEIGHT_DATA,
    "vocab.json": b"{}",
    "source.spm": b"source vocabulary",
    "target.spm": b"target vocabulary",
}


@contextmanager
def _model_server(mode, repo="Mitsua/elan-mt-bt-ja-en", files=None):
    files = files or _MITSUA_FILES
    offsets = []

    class Server(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == f"/api/models/{repo}/revision/main?blobs=true":
                body = json.dumps({
                    "sha": _REVISION,
                    "siblings": [{
                        "rfilename": name,
                        "size": len(data),
                        "lfs": {"sha256": sha256(data).hexdigest()},
                    } for name, data in files.items()],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            prefix = f"/{repo}/resolve/{_REVISION}/"
            name = self.path.removeprefix(prefix)
            if name == self.path or name not in files:
                self.send_error(404)
                return

            if mode != "optional-file" and name != "pytorch_model.bin":
                data = files[name]
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            offset = self.headers.get("Range")
            offsets.append(offset)
            if mode == "optional-file":
                data = files[name]
                body = data[:1] if name == "generation_config.json" else data
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(body)
            elif mode == "corrupt":
                self.send_response(200)
                self.send_header("Content-Length", str(len(_WEIGHT_DATA)))
                self.end_headers()
                self.wfile.write(b"x" * len(_WEIGHT_DATA))
            elif len(offsets) == 1:
                self.send_response(200)
                self.send_header("Content-Length", str(len(_WEIGHT_DATA)))
                self.end_headers()
                self.wfile.write(_WEIGHT_DATA[:12])
            else:
                start = 12 if mode == "resume" else 0
                body = _WEIGHT_DATA[start:]
                self.send_response(206 if start else 200)
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(_WEIGHT_DATA) - 1}/{len(_WEIGHT_DATA)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Server) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", offsets
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("mode", ["resume", "range-ignored"])
def test_model_download_preserves_a_verified_resume_candidate_and_reports_bytes(tmp_path, monkeypatch, mode):
    from core.setup_types import DownloadInterrupted

    snapshot = tmp_path / "models" / "hub" / "models--Mitsua--elan-mt-bt-ja-en" / "snapshots" / _REVISION
    weight = snapshot / "pytorch_model.bin"
    prefix_bytes = len(_MITSUA_FILES["config.json"])
    with _model_server(mode) as (endpoint, offsets):
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        item = _items(tmp_path)["translate-ja-en"]
        events = []
        with pytest.raises(DownloadInterrupted):
            item.install(lambda completed, total: events.append((completed, total)))
        assert not item.is_ready()
        assert not weight.exists()
        assert events[-1] == (prefix_bytes + 12, sum(map(len, _MITSUA_FILES.values())))

        retry_progress = []
        item.install(lambda completed, total: retry_progress.append((completed, total)))

    assert offsets == [None, "bytes=12-"]
    assert (prefix_bytes + (12 if mode == "resume" else 0), sum(map(len, _MITSUA_FILES.values()))) in retry_progress
    assert retry_progress[-1] == (sum(map(len, _MITSUA_FILES.values())),) * 2
    assert weight.read_bytes() == _WEIGHT_DATA
    assert item.is_ready()
    assert not list((tmp_path / "models").rglob("*.part"))


def test_same_size_corrupt_download_is_never_published(tmp_path, monkeypatch):
    with _model_server("corrupt") as (endpoint, _):
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        item = _items(tmp_path)["translate-ja-en"]
        with pytest.raises(RuntimeError, match="checksum"):
            item.install(lambda _completed, _total: None)
    assert not item.is_ready()
    assert not list((tmp_path / "models").rglob("*.part"))


def test_invalid_endpoint_does_not_expose_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://user:secret@127.0.0.1")
    with pytest.raises(RuntimeError, match="model server address") as error:
        _items(tmp_path)["translate-ja-en"].install(lambda *_: None)
    assert "secret" not in str(error.value)


def test_interruption_during_last_asset_keeps_snapshot_unready(tmp_path, monkeypatch):
    from core.setup_types import DownloadInterrupted

    with _model_server("optional-file", "Mitsua/elan-mt-bt-ja-en", {
        "config.json": b"{}",
        "pytorch_model.bin": b"weight data",
        "vocab.json": b"{}",
        "source.spm": b"source vocabulary",
        "target.spm": b"target vocabulary",
        "generation_config.json": b"{}",
    }) as (endpoint, _):
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        item = _items(tmp_path)["translate-ja-en"]
        with pytest.raises(DownloadInterrupted):
            item.install(lambda _completed, _total: None)
        assert not item.is_ready()
def test_cache_path_supports_dotenv_default(tmp_path):
    from core.setup_environment import read_setup_environment

    (tmp_path / ".env").write_text(
        'export HF_HOME="${CACHE_DIR:-weights}/hub-cache" # local\n',
        encoding="utf-8-sig",
    )
    assert read_setup_environment(tmp_path)["HF_HOME"] == "weights/hub-cache"
