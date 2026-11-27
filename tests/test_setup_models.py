from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import ssl
from threading import Thread
from urllib.error import URLError

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


def test_startup_lists_addons_separately_from_required_models(tmp_path, monkeypatch):
    from core.setup_models import model_items

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    items = model_items(tmp_path, include_optional=True)
    assert {item.key for item in items if item.optional} == {
        "translate-ja-en-nllb", "translate-ja-vi-nllb-custom", "live-ocr-medium-det", "live-ocr-medium-rec",
        "live-translate-qwen", "live-translate-qwen-2b", "live-qwen-runtime",
        "live-translate-qwen-9b", "live-translate-qwen-27b", "live-qwen-cuda-runtime",
    }
    assert {item.key for item in items if not item.optional} == {
        "translate-ja-en", "live-translate-en-vi", "live-ocr-det", "live-ocr-rec",
    }


def test_direct_inventory_requires_qwen_and_its_runtime_instead_of_pivot_models(tmp_path):
    from core.ocr_settings import PRESETS
    from core.setup_models import available_model_settings, model_items

    settings = PRESETS["Ultra"]
    settings.save(tmp_path)
    direct = {"live-ocr-medium-det", "live-ocr-medium-rec", "live-translate-qwen", "live-qwen-runtime"}
    assert {item.key for item in model_items(tmp_path, settings)} == direct
    startup = model_items(tmp_path, include_optional=True)
    assert {item.key for item in startup if not item.optional} == direct
    assert available_model_settings(tmp_path) == settings



@pytest.mark.parametrize("variant,key,filename", [
    ("2b", "live-translate-qwen-2b", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("4b", "live-translate-qwen", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("9b", "live-translate-qwen-9b", "Qwen3.5-9B-Q4_K_M.gguf"),
    ("27b", "live-translate-qwen-27b", "Qwen3.8-27B-Q4_K_M.gguf"),
])
def test_direct_inventory_selects_only_requested_qwen_variant(tmp_path, variant, key, filename):
    from dataclasses import replace
    from core.ocr_settings import PRESETS
    from core.setup_models import model_items, qwen_model_path

    settings = replace(PRESETS["Ultra"], qwen_model=variant)
    selected = model_items(tmp_path, settings)
    assert {item.key for item in selected if item.key.startswith("live-translate-qwen")} == {key}
    assert not {"translate-ja-en", "live-translate-en-vi"} & {item.key for item in selected}
    all_items = model_items(tmp_path, settings, include_optional=True)
    qwen = {item.key: item for item in all_items if item.key.startswith("live-translate-qwen")}
    assert set(qwen) == {
        "live-translate-qwen", "live-translate-qwen-2b",
        "live-translate-qwen-9b", "live-translate-qwen-27b",
    }
    assert not qwen[key].optional
    assert all(item.optional for other, item in qwen.items() if other != key)
    assert qwen_model_path(tmp_path, variant) == tmp_path / "model" / filename


def test_qwen_runtime_installs_from_verified_zip_and_detects_missing_dll(tmp_path, monkeypatch):
    from zipfile import ZipFile

    from core import setup_models

    archive = setup_models._qwen_runtime_archive(tmp_path)
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, "w") as bundle:
        for name in setup_models._RUNTIME_REQUIRED:
            bundle.writestr(name, b"binary data")
        bundle.writestr("../outside.exe", b"unused and outside the runtime directory")
    monkeypatch.setattr(
        setup_models, "_QWEN_RUNTIME_ASSET",
        setup_models._RemoteFile(archive.stat().st_size, sha256(archive.read_bytes()).hexdigest()),
    )
    inventory = {item.key: item for item in setup_models.model_items(tmp_path, include_optional=True)}
    runtime = inventory["live-qwen-runtime"]
    assert not runtime.is_ready()
    runtime.install(lambda _completed, _total: None)
    assert runtime.is_ready()
    assert not (setup_models.qwen_server_path(tmp_path).parent.parent / "outside.exe").exists()
    setup_models.qwen_server_path(tmp_path).with_name("ggml-vulkan.dll").unlink()
    assert not runtime.is_ready()


def test_cuda_backend_requires_only_its_runtime_and_remains_optional_for_vulkan(tmp_path):
    from dataclasses import replace

    from core.ocr_settings import PRESETS
    from core.setup_models import model_items

    default = PRESETS["Ultra"]
    selected = replace(default, qwen_backend="cuda-python")
    default_items = {item.key: item for item in model_items(tmp_path, default, include_optional=True)}
    assert default_items["live-qwen-cuda-runtime"].optional
    assert not default_items["live-qwen-runtime"].optional
    direct = {item.key for item in model_items(tmp_path, selected)}
    assert "live-qwen-cuda-runtime" in direct
    assert "live-qwen-runtime" not in direct
    all_items = {item.key: item for item in model_items(tmp_path, selected, include_optional=True)}
    assert all_items["live-qwen-runtime"].optional
    assert not all_items["live-qwen-cuda-runtime"].optional


def test_cuda_runtime_installs_avx2_dlls_in_private_package(tmp_path, monkeypatch):
    from zipfile import ZipFile

    from core import setup_libraries, setup_models

    directory = setup_models.qwen_cuda_runtime_path(tmp_path)
    wheel = directory.parent / setup_models._QWEN_CUDA_WHEEL
    archive = directory.parent / setup_models._QWEN_CUDA_CPU_ZIP
    _asset(wheel, b"local wheel fixture")
    expected = {name: name.encode() for name in setup_models._QWEN_CUDA_CPU_FILES}
    with ZipFile(archive, "w") as bundle:
        for name, data in expected.items():
            bundle.writestr(name, data)
        bundle.writestr("../outside.dll", b"ignored")
    for key, asset in (("_QWEN_CUDA_WHEEL_ASSET", wheel), ("_QWEN_CUDA_CPU_ASSET", archive)):
        data = asset.read_bytes()
        monkeypatch.setattr(setup_models, key, setup_models._RemoteFile(len(data), sha256(data).hexdigest()))

    def fake_pip(executable, args, root, progress, label, log):
        package = Path(args[args.index("--target") + 1])
        assert not package.is_relative_to(directory) and "--no-deps" in args and str(wheel) in args
        for name in setup_models._QWEN_CUDA_REQUIRED:
            if Path(name).name not in {"libomp.dll", "LICENSE-LLVM-OpenMP"}:
                _asset(package.parent / name, b"original from wheel")

    monkeypatch.setattr(setup_libraries, "_pip", fake_pip)
    items = {item.key: item for item in setup_models.model_items(tmp_path, include_optional=True)}
    item = items["live-qwen-cuda-runtime"]
    assert not item.is_ready()
    item.install(lambda _completed, _total: None)
    assert item.is_ready()
    native = directory / "python" / "llama_cpp" / "lib"
    assert (native / "ggml-cpu.dll").read_bytes() == expected["ggml-cpu-haswell.dll"]
    assert (native / "libomp.dll").read_bytes() == expected["libomp.dll"]
    assert not list(tmp_path.rglob("outside.dll"))
    (native / "ggml-cuda.dll").unlink()
    assert not item.is_ready()


def test_missing_saved_addons_fall_back_without_overwriting_saved_selection(tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from core.setup_models import available_model_settings

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    saved = OCRSettings("medium", "medium", "nllb")
    saved.save(tmp_path)
    installed = tmp_path / "models" / "live-ocr" / "PP-OCRv6_medium_det_safetensors"
    for name in ("config.json", "preprocessor_config.json", "inference.yml", "model.safetensors"):
        _asset(installed / name, b"{}")
    # The existing model tests cover the weight format. This case needs a valid tiny container.
    header = b'{"weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}'
    _asset(installed / "model.safetensors", len(header).to_bytes(8, "little") + header + b"\0\0")
    assert available_model_settings(tmp_path) == OCRSettings("medium", "small", "mitsua")
    assert OCRSettings.load(tmp_path) == saved


def test_startup_scans_saved_combination_and_settings_can_scan_another(tmp_path):
    from core.ocr_settings import OCRSettings
    from core.setup_models import model_items, resolve_model

    settings = OCRSettings("medium", "small")
    settings.save(tmp_path)
    assert OCRSettings.load(tmp_path) == settings
    assert set(_items(tmp_path)) == {
        "translate-ja-en", "live-translate-en-vi", "live-ocr-medium-det", "live-ocr-rec",
    }
    assert resolve_model(tmp_path, "live-ocr-medium-det").endswith("PP-OCRv6_medium_det_safetensors")
    assert [item.key for item in model_items(tmp_path, OCRSettings("small", "medium")) if item.key.startswith("live-ocr")] == [
        "live-ocr-det", "live-ocr-medium-rec",
    ]


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


def test_medium_requires_custom_onnx_instead_of_japanese_to_english_weights(tmp_path):
    from core.ocr_settings import PRESETS
    from core.setup_models import model_items, resolve_model

    key = "translate-ja-vi-nllb-custom"
    items = {item.key for item in model_items(tmp_path, PRESETS["Medium"])}
    assert items == {key, "live-translate-en-vi", "live-ocr-medium-det", "live-ocr-rec"}
    assert resolve_model(tmp_path, key) == "belumind/nllb-trilingual-en-vi-ja-onnx"


def test_saved_nllb_selection_scans_its_own_checkpoint(tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from core.setup_models import resolve_model

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    old_settings = tmp_path / "models" / "live-ocr" / "settings.json"
    _asset(old_settings, b'{"det":"medium","rec":"small"}')
    assert "translate-ja-en" in _items(tmp_path)

    OCRSettings("medium", "small", "nllb").save(tmp_path)
    key = "translate-ja-en-nllb"
    source = "facebook/nllb-200-distilled-600M"
    assert set(_items(tmp_path)) == {key, "live-translate-en-vi", "live-ocr-medium-det", "live-ocr-rec"}
    assert resolve_model(tmp_path, key) == source
    snapshot = tmp_path / "models" / "hub" / "models--facebook--nllb-200-distilled-600M" / "snapshots" / ("a" * 40)
    _asset(snapshot / "config.json", b"{}")
    _asset(snapshot / "pytorch_model.bin")
    assert not _items(tmp_path)[key].is_ready()
    _asset(snapshot / "tokenizer.json", b"{}")
    assert _items(tmp_path)[key].is_ready()
    assert resolve_model(tmp_path, key) == str(snapshot)


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


def _certificate_error(code, message):
    error = ssl.SSLCertVerificationError(1, message)
    error.verify_code = code
    error.verify_message = message
    return URLError(error)


def test_model_request_retries_missing_issuer_with_verified_public_cas(monkeypatch):
    from core import setup_models

    response = object()
    calls = []

    def open_url(request, timeout, context=None):
        calls.append((request, timeout, context))
        if context is None:
            raise _certificate_error(20, "unable to get local issuer certificate")
        return response

    monkeypatch.setattr(setup_models, "urlopen", open_url)
    result = setup_models._request({}, "https://github.com/model", Range="bytes=32-")

    assert result is response
    assert len(calls) == 2
    first, retry = calls
    assert retry[:2] == first[:2]
    assert retry[0].get_header("Range") == "bytes=32-"
    assert retry[2].check_hostname and retry[2].verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize(("code", "message", "attempts"), [
    (20, "unable to get local issuer certificate", 2),
    (10, "certificate has expired", 1),
    (62, "hostname mismatch", 1),
])
def test_model_request_reports_certificate_failures_without_network_retry(monkeypatch, code, message, attempts):
    from core import setup_models
    from core.setup_types import DownloadInterrupted

    calls = []

    def open_url(request, timeout, context=None):
        calls.append(context)
        raise _certificate_error(code, message)

    monkeypatch.setattr(setup_models, "urlopen", open_url)
    with pytest.raises(RuntimeError, match=f"TLS certificate.*github.com.*{message}") as failure:
        setup_models._request({}, "https://github.com/model")

    assert not isinstance(failure.value, DownloadInterrupted)
    assert len(calls) == attempts


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
