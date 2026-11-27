"""Offline model inventory and explicit, verified downloads for MangaLens."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
from shutil import copyfileobj
import ssl
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

from core.ocr_settings import (
    DEFAULT_JA_MODEL, FLOW_DIRECT, FLOW_NLLB_VINAI, JA_MODEL_KEYS, OCR_SIZES,
    QWEN_2B, QWEN_4B, QWEN_9B, QWEN_27B, QWEN_BACKEND_CUDA_PYTHON, QWEN_MODEL_LABELS,
    OCRSettings, ocr_model_key,
)
from core.setup_environment import read_setup_environment
from core.setup_types import DownloadCertificateError, DownloadInterrupted, InstallationItem, InstallLog, Progress


_MITSUA = "Mitsua/elan-mt-bt-ja-en"
_NLLB = "facebook/nllb-200-distilled-600M"
_NLLB_CUSTOM = "belumind/nllb-trilingual-en-vi-ja-onnx"
NLLB_CUSTOM_MODEL_KEY = "translate-ja-vi-nllb-custom"
NLLB_ONNX_FILES = (
    "encoder_model_quantized.onnx", "decoder_model_quantized.onnx", "decoder_with_past_model_quantized.onnx",
)
_VINAI = "vinai/vinai-translate-en2vi-v2"
QWEN_MODEL_KEY = "live-translate-qwen"
QWEN_2B_MODEL_KEY = "live-translate-qwen-2b"
QWEN_9B_MODEL_KEY = "live-translate-qwen-9b"
QWEN_27B_MODEL_KEY = "live-translate-qwen-27b"
QWEN_RUNTIME_KEY = "live-qwen-runtime"
QWEN_CUDA_RUNTIME_KEY = "live-qwen-cuda-runtime"
_QWEN_MODEL = "Qwen3.5-4B-Q4_K_M.gguf"
_QWEN_MODEL_URL = f"https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF/resolve/main/{_QWEN_MODEL}"
_QWEN_RUNTIME_RELEASE = "b12621"
_QWEN_RUNTIME_ZIP = f"llama-{_QWEN_RUNTIME_RELEASE}-bin-win-vulkan-x64.zip"
_QWEN_RUNTIME_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{_QWEN_RUNTIME_RELEASE}/{_QWEN_RUNTIME_ZIP}"
_RUNTIME_REQUIRED = {
    "llama-server.exe", "llama-server-impl.dll", "llama-common.dll", "llama.dll",
    "ggml-base.dll", "ggml-vulkan.dll", "ggml.dll", "mtmd.dll",
}
_QWEN_CUDA_RELEASE = "0.3.37-cu132"
_QWEN_CUDA_WHEEL = "llama_cpp_python-0.3.37-py3-none-win_amd64.whl"
_QWEN_CUDA_WHEEL_URL = (
    f"https://github.com/abetlen/llama-cpp-python/releases/download/v{_QWEN_CUDA_RELEASE}/{_QWEN_CUDA_WHEEL}"
)
# b12387 is the llama.cpp commit included in the Python wheel. The official
# wheel's CPU DLL uses AVX-512, so deploy the matching upstream AVX2 variant.
_QWEN_CUDA_CPU_ZIP = "llama-b12387-bin-win-cpu-x64.zip"
_QWEN_CUDA_CPU_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/b12387/{_QWEN_CUDA_CPU_ZIP}"
_QWEN_CUDA_CPU_FILES = {
    "ggml-cpu-haswell.dll": "ggml-cpu.dll",
    "libomp.dll": "libomp.dll",
    "LICENSE-LLVM-OpenMP": "LICENSE-LLVM-OpenMP",
}
_QWEN_CUDA_REQUIRED = (
    "python/llama_cpp/__init__.py",
    "python/llama_cpp/lib/llama.dll",
    "python/llama_cpp/lib/mtmd.dll",
    "python/llama_cpp/lib/ggml.dll",
    "python/llama_cpp/lib/ggml-base.dll",
    "python/llama_cpp/lib/ggml-cpu.dll",
    "python/llama_cpp/lib/ggml-cuda.dll",
    "python/llama_cpp/lib/libomp.dll",
    "python/llama_cpp/lib/LICENSE-LLVM-OpenMP",
    "python/diskcache/__init__.py",
    "python/jinja2/__init__.py",
    "python/markupsafe/__init__.py",
    "python/typing_extensions.py",
)
_QWEN_CUDA_LABEL = "llama-cpp-python CUDA (Qwen)"
_OPTIONAL_MODEL_KEYS = {
    JA_MODEL_KEYS["nllb"], ocr_model_key("det", "medium"), ocr_model_key("rec", "medium"),
}
_CHUNK_SIZE = 1024 * 1024
_PENDING_NAME = ".mangalens-setup.pending"
_FILENAME = re.compile(r"[\w.-]+", re.ASCII)
_REVISION = re.compile(r"[a-f0-9]{40}")
_WEIGHTS = (("model.safetensors",), ("model.safetensors.index.json",),
            ("pytorch_model.bin",), ("pytorch_model.bin.index.json",))
_MARIAN_TOKEN = (("vocab.json", "source.spm", "target.spm"), ("tokenizer.json",))
_MBART_TOKEN = (("sentencepiece.bpe.model",), ("tokenizer.json",))


@dataclass(frozen=True)
class _Files:
    required: tuple[str, ...]
    alternatives: tuple[tuple[tuple[str, ...], ...], ...] = ()
    optional: tuple[str, ...] = ()


_PROFILES = {
    "marian": _Files(("config.json",), (_WEIGHTS, _MARIAN_TOKEN), (
        "generation_config.json", "tokenizer_config.json", "special_tokens_map.json",
    )),
    "mbart": _Files(("config.json",), (_WEIGHTS, _MBART_TOKEN), (
        "generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    )),
    "nllb": _Files(("config.json",), (_WEIGHTS, (("tokenizer.json",), ("sentencepiece.bpe.model",))), (
        "generation_config.json", "tokenizer_config.json", "special_tokens_map.json",
    )),
    "nllb-onnx": _Files(
        ("config.json", "tokenizer.json", "tokenizer_config.json", *NLLB_ONNX_FILES),
        optional=("special_tokens_map.json",),
    ),
    "live-ocr": _Files(("config.json", "preprocessor_config.json", "inference.yml", "model.safetensors")),
}


@dataclass(frozen=True)
class _Model:
    key: str
    label: str
    source: str
    files: _Files
    target: Path | None = None


@dataclass(frozen=True)
class _RemoteFile:
    size: int
    checksum: str


_QWEN_MODEL_ASSET = _RemoteFile(3032091584, "217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14")
_QWEN_2B_MODEL_ASSET = _RemoteFile(1312165056, "a8b50de00b780c0f3c16f07cb041a1ecd678b6e22fd8cde91e57fe4b5031523b")
_QWEN_9B_MODEL_ASSET = _RemoteFile(5680522464, "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8")
_QWEN_27B_MODEL_ASSET = _RemoteFile(19403684480, "d15d4b6de55f662ac40e867702d31afe738d032fa06b829b2b1a21a49d4a48ee")
_QWEN_VARIANTS = {
    QWEN_4B: (QWEN_MODEL_KEY, _QWEN_MODEL, _QWEN_MODEL_URL, _QWEN_MODEL_ASSET),
    QWEN_2B: (QWEN_2B_MODEL_KEY, "Qwen3.5-2B-Q4_K_M.gguf",
              "https://huggingface.co/SoAIHQ/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf",
              _QWEN_2B_MODEL_ASSET),
    QWEN_9B: (QWEN_9B_MODEL_KEY, "Qwen3.5-9B-Q4_K_M.gguf",
              "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/"
              "Qwen3.5-9B-Q4_K_M.gguf",
              _QWEN_9B_MODEL_ASSET),
    QWEN_27B: (QWEN_27B_MODEL_KEY, "Qwen3.8-27B-Q4_K_M.gguf",
               "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/c1869413dece09109e55b7dcc2c77ef382b0d0bd/"
               "Qwen3.8-27B-Q4_K_M.gguf",
               _QWEN_27B_MODEL_ASSET),
}
_QWEN_RUNTIME_ASSET = _RemoteFile(38871144, "38c58fe8b3eb5bfb892d02e9b67349a1ee8eb5d1bf81c0fb4894a2d7ebaf1e4d")
_QWEN_CUDA_WHEEL_ASSET = _RemoteFile(261129399, "ba1afbd5b7f5010cdb4c3312a504eac7e562b7aae34c8d16c4f5b7e1d761ecab")
_QWEN_CUDA_CPU_ASSET = _RemoteFile(25124944, "ced8482efac76c5f24398b97a6fe1662109038ec3b880686fd7c7c8f22bc5d99")


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _hub_caches(root: Path, env: dict[str, str]) -> tuple[Path, ...]:
    home = _path(root, env.get("HF_HOME") or "models")
    current = _path(root, env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE") or home / "hub")
    user_cache = _path(root, env.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    # The singular directory was used by earlier MangaLens versions.
    return tuple(dict.fromkeys((
        current, _path(root, env.get("TRANSFORMERS_CACHE") or current),
        root / "models" / "hub", root / "model" / "hub", user_cache / "huggingface" / "hub",
    )))


def _model_specs(root: Path) -> dict[str, _Model]:
    models = [
        _Model(JA_MODEL_KEYS[DEFAULT_JA_MODEL], "Japanese → English (Mitsua ElanMT)", _MITSUA, _PROFILES["marian"]),
        _Model("live-translate-en-vi", "English → Vietnamese (VinAI Translate v2)", _VINAI, _PROFILES["mbart"]),
        _Model(JA_MODEL_KEYS["nllb"], "Japanese → English (NLLB-200 Distilled 600M)", _NLLB, _PROFILES["nllb"]),
        _Model(NLLB_CUSTOM_MODEL_KEY, "Japanese → Vietnamese (NLLB Custom ONNX)", _NLLB_CUSTOM,
               _PROFILES["nllb-onnx"]),
    ]
    for size in OCR_SIZES:
        for role in ("det", "rec"):
            name = f"PP-OCRv6_{size}_{role}_safetensors"
            models.append(_Model(
                ocr_model_key(role, size), f"Live OCR ({name})", f"PaddlePaddle/{name}",
                _PROFILES["live-ocr"], root / "models" / "live-ocr" / name,
            ))
    return {model.key: model for model in models}


def _file_present(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(128)
            if not header or header.startswith(b"version https://git-lfs.github.com/spec/") or header.lstrip().lower().startswith((b"<html", b"<!doctype html")):
                return False
            if path.suffix == ".json":
                stream.seek(0)
                return isinstance(json.load(stream), dict)
            if path.suffix == ".safetensors":
                header_size = int.from_bytes(header[:8], "little")
                size = path.stat().st_size
                if not 0 < header_size <= min(size - 8, 100_000_000):
                    return False
                stream.seek(8)
                tensors = json.loads(stream.read(header_size))
                offsets = [tensor["data_offsets"][1] for key, tensor in tensors.items() if key != "__metadata__"]
                return bool(offsets) and size == 8 + header_size + max(offsets)
            if path.suffix in {".bin", ".pt"} and header.startswith(b"PK\x03\x04"):
                return is_zipfile(stream)
            return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


def _index_shards(path: Path) -> tuple[str, ...]:
    try:
        weight_map = json.loads(path.read_bytes())["weight_map"]
        names = tuple(dict.fromkeys(weight_map.values()))
        if names and all(_FILENAME.fullmatch(name)
                         and name.endswith((".safetensors", ".bin")) for name in names):
            return names
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return ()


def _has_files(directory: Path, files: _Files) -> bool:
    if (directory / _PENDING_NAME).exists():
        return False

    def present(name):
        path = directory / name
        if not _file_present(path):
            return False
        if name.endswith(".index.json"):
            shards = _index_shards(path)
            return bool(shards) and all(_file_present(directory / shard) for shard in shards)
        return True

    return all(map(present, files.required)) and all(
        any(all(map(present, choice)) for choice in alternatives) for alternatives in files.alternatives
    )


def _cache_repo(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def _snapshots(cache: Path, source: str):
    repo = _cache_repo(cache, source)
    try:
        current = (repo / "refs" / "main").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        current = ""
    if _REVISION.fullmatch(current):
        yield repo / "snapshots" / current
    for snapshot in sorted((repo / "snapshots").glob("*"), reverse=True):
        if snapshot.name != current and _REVISION.fullmatch(snapshot.name):
            yield snapshot


def _ready_path(root: Path, env: dict[str, str], model: _Model) -> Path | None:
    if model.target is not None and _has_files(model.target, model.files):
        return model.target
    for cache in _hub_caches(root, env):
        for location in _snapshots(cache, model.source):
            if _has_files(location, model.files):
                return location
    return None


def resolve_model(root: Path, key: str) -> str:
    """Return the same local model that setup checked, or the configured source."""
    root = root.resolve()
    env = read_setup_environment(root)
    model = _model_specs(root)[key]
    found = _ready_path(root, env, model)
    if found is not None:
        return str(found)
    if model.target is not None:
        return str(model.target)
    return model.source


def qwen_model_path(root: Path, variant: str = QWEN_4B) -> Path:
    return root / "model" / _QWEN_VARIANTS[variant][1]


def qwen_server_path(root: Path) -> Path:
    return root / "models" / "live-qwen-runtime" / f"{_QWEN_RUNTIME_RELEASE}-vulkan" / "llama-server.exe"


def qwen_cuda_runtime_path(root: Path) -> Path:
    return root / "models" / "live-qwen-runtime" / f"llama-cpp-python-{_QWEN_CUDA_RELEASE}-avx2"


def qwen_cuda_runtime_ready(root: Path) -> bool:
    directory = qwen_cuda_runtime_path(root)
    try:
        files = json.loads((directory / "installed.json").read_text(encoding="utf-8"))
        return isinstance(files, dict) and all(
            type(files.get(name)) is int and files[name] > 0 and _same_size(directory / name, files[name])
            for name in _QWEN_CUDA_REQUIRED
        )
    except (OSError, ValueError):
        return False


def _qwen_runtime_archive(root: Path) -> Path:
    return qwen_server_path(root).parent.parent / _QWEN_RUNTIME_ZIP


def _qwen_model_ready(root: Path, variant: str = QWEN_4B) -> bool:
    path = qwen_model_path(root, variant)
    if not _same_size(path, _QWEN_VARIANTS[variant][3].size):
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"GGUF"
    except OSError:
        return False


def _qwen_runtime_ready(root: Path) -> bool:
    directory = qwen_server_path(root).parent
    try:
        files = json.loads((directory / "installed.json").read_text(encoding="utf-8"))
        return (
            isinstance(files, dict) and _RUNTIME_REQUIRED.issubset(files)
            and all(
                _FILENAME.fullmatch(name) and name not in {".", ".."}
                and type(size) is int and size > 0 and _same_size(directory / name, size)
                for name, size in files.items()
            )
        )
    except (OSError, ValueError):
        return False


def _install_qwen_model(root: Path, progress: Progress, variant: str = QWEN_4B) -> None:
    if not _qwen_model_ready(root, variant):
        _, _, url, asset = _QWEN_VARIANTS[variant]
        _download({}, url, asset, qwen_model_path(root, variant), lambda amount: progress(amount, asset.size))
        if not _qwen_model_ready(root, variant):
            raise RuntimeError("The verified Qwen file is not a complete GGUF model.")


def _install_qwen_runtime(root: Path, progress: Progress) -> None:
    if _qwen_runtime_ready(root):
        return
    archive = _qwen_runtime_archive(root)
    asset = _QWEN_RUNTIME_ASSET
    _download({}, _QWEN_RUNTIME_URL, asset, archive, lambda amount: progress(amount, asset.size))
    directory = qwen_server_path(root).parent
    with TemporaryDirectory(dir=directory.parent, prefix="extract-") as temporary, ZipFile(archive) as bundle:
        stage = Path(temporary)
        files = {}
        for member in bundle.infolist():
            name = member.filename
            if not _FILENAME.fullmatch(name) or not (name.endswith(".dll") or name == "llama-server.exe"):
                continue
            with bundle.open(member) as source, (stage / name).open("xb") as destination:
                copyfileobj(source, destination)
            files[name] = member.file_size
        if not _RUNTIME_REQUIRED.issubset(files):
            raise RuntimeError("The llama.cpp archive is missing files required by its server.")
        (stage / "installed.json").write_text(json.dumps(files), encoding="utf-8")
        directory.mkdir(exist_ok=True)
        manifest = directory / "installed.json"
        manifest.unlink(missing_ok=True)
        for name in files:
            (stage / name).replace(directory / name)
        (stage / manifest.name).replace(manifest)


def _install_qwen_cuda_runtime(root: Path, progress: Progress, log: InstallLog | None = None) -> None:
    from core.setup_libraries import _live_executable, _pip, _replace_runtime

    if qwen_cuda_runtime_ready(root):
        return
    directory = qwen_cuda_runtime_path(root)
    wheel = directory.parent / _QWEN_CUDA_WHEEL
    archive = directory.parent / _QWEN_CUDA_CPU_ZIP
    wheel_size = _QWEN_CUDA_WHEEL_ASSET.size
    total = wheel_size + _QWEN_CUDA_CPU_ASSET.size
    _download({}, _QWEN_CUDA_WHEEL_URL, _QWEN_CUDA_WHEEL_ASSET, wheel, lambda amount: progress(amount, total))
    _download({}, _QWEN_CUDA_CPU_URL, _QWEN_CUDA_CPU_ASSET, archive, lambda amount: progress(wheel_size + amount, total))

    executable = _live_executable(root, read_setup_environment(root))
    with TemporaryDirectory(dir=directory.parent, prefix=".qwen-cuda-") as temporary:
        stage = Path(temporary) / "runtime"
        package = stage / "python"
        stage.mkdir()
        _pip(
            executable, [
                "--target", str(package), "--only-binary=:all:", "--no-deps", str(wheel),
                "diskcache==5.6.3", "Jinja2==3.1.6", "MarkupSafe==3.0.4", "typing-extensions==4.16.0",
            ], root, progress, _QWEN_CUDA_LABEL, log,
        )
        native = package / "llama_cpp" / "lib"
        with ZipFile(archive) as bundle:
            for source_name, target_name in _QWEN_CUDA_CPU_FILES.items():
                with bundle.open(source_name) as source, (native / target_name).open("wb") as target:
                    copyfileobj(source, target)
        files = {}
        for name in _QWEN_CUDA_REQUIRED:
            file = stage / name
            if not file.is_file() or (size := file.stat().st_size) == 0:
                raise RuntimeError(f"Incomplete {_QWEN_CUDA_LABEL} installation: missing {name}.")
            files[name] = size
        (stage / "installed.json").write_text(json.dumps(files), encoding="utf-8")
        _replace_runtime(directory, stage)
        progress(total, total)


def model_items(
    root: Path, ocr_settings: OCRSettings | None = None, *, include_optional: bool = False,
) -> list[InstallationItem]:
    """Return selected models or the startup inventory with its optional add-ons."""
    root = root.resolve()
    settings = ocr_settings or OCRSettings.load(root)
    if include_optional and settings.flow != FLOW_DIRECT:
        settings = OCRSettings()
    selected = {
        ocr_model_key("det", settings.det), ocr_model_key("rec", settings.rec),
    }
    if settings.flow == FLOW_DIRECT:
        runtime_key = QWEN_CUDA_RUNTIME_KEY if settings.qwen_backend == QWEN_BACKEND_CUDA_PYTHON else QWEN_RUNTIME_KEY
        selected.update({_QWEN_VARIANTS[settings.qwen_model][0], runtime_key})
    else:
        japanese_key = NLLB_CUSTOM_MODEL_KEY if settings.flow == FLOW_NLLB_VINAI else JA_MODEL_KEYS[settings.ja_model]
        selected.update({japanese_key, "live-translate-en-vi"})

    def check(model):
        return _ready_path(root, read_setup_environment(root), model) is not None

    def install(model, progress):
        env = read_setup_environment(root)
        if _ready_path(root, env, model) is None:
            _install(root, env, model, progress)

    items = [
        InstallationItem(
            model.key, model.label, "model",
            lambda model=model: check(model), lambda progress, model=model: install(model, progress),
            optional=model.key not in selected,
        )
        for model in _model_specs(root).values()
        if model.key in selected or include_optional
    ]
    for variant, (key, _, _, _) in _QWEN_VARIANTS.items():
        if key in selected or include_optional:
            items.append(InstallationItem(
                key, QWEN_MODEL_LABELS[variant], "model", partial(_qwen_model_ready, root, variant),
                partial(_install_qwen_model, root, variant=variant),
                optional=key not in selected,
            ))
    if QWEN_RUNTIME_KEY in selected or include_optional:
        items.append(InstallationItem(
            QWEN_RUNTIME_KEY, "llama.cpp Vulkan (Qwen)", "model", partial(_qwen_runtime_ready, root),
            partial(_install_qwen_runtime, root), optional=QWEN_RUNTIME_KEY not in selected,
        ))
    if QWEN_CUDA_RUNTIME_KEY in selected or include_optional:
        installer = partial(_install_qwen_cuda_runtime, root)
        items.append(InstallationItem(
            QWEN_CUDA_RUNTIME_KEY, _QWEN_CUDA_LABEL, "model", partial(qwen_cuda_runtime_ready, root),
            installer, installer, optional=QWEN_CUDA_RUNTIME_KEY not in selected,
        ))
    return items


def available_model_settings(root: Path) -> OCRSettings:
    """Use the defaults for selected add-ons whose local files are missing."""
    settings = OCRSettings.load(root)
    if settings.flow in (FLOW_DIRECT, FLOW_NLLB_VINAI):
        return settings
    missing = {
        item.key for item in model_items(root, settings)
        if item.key in _OPTIONAL_MODEL_KEYS and not item.is_ready()
    }
    return replace(
        settings,
        det="small" if ocr_model_key("det", settings.det) in missing else settings.det,
        rec="small" if ocr_model_key("rec", settings.rec) in missing else settings.rec,
        ja_model=DEFAULT_JA_MODEL if JA_MODEL_KEYS[settings.ja_model] in missing else settings.ja_model,
    )


def _open_model_url(request):
    try:
        return urlopen(request, timeout=30)
    except (URLError, ssl.SSLCertVerificationError) as error:
        reason = getattr(error, "reason", error)
        # OpenSSL codes for missing issuers or an incomplete certificate chain.
        if isinstance(reason, ssl.SSLCertVerificationError) and reason.verify_code in {2, 20, 21}:
            try:
                import certifi
            except ImportError:
                pass  # Setup may run before the user has installed its libraries.
            else:
                context = ssl.create_default_context()
                try:
                    context.load_verify_locations(cafile=certifi.where())
                except OSError as ca_error:
                    raise DownloadCertificateError("Could not load the certifi CA bundle.") from ca_error
                return urlopen(request, timeout=30, context=context)
        raise


def _request(env, url, **headers):
    request = Request(url, headers={"Accept-Encoding": "identity", **headers})
    if token := env.get("HF_TOKEN"):
        if not token.isascii() or not token.isprintable() or " " in token:
            raise RuntimeError("The configured Hugging Face token has an invalid format.")
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return _open_model_url(request)
    except HTTPError as error:
        error.close()
        if error.code in {408, 425, 429, 500, 502, 503, 504}:
            raise DownloadInterrupted(f"Model server temporarily returned HTTP {error.code}.") from error
        raise RuntimeError(f"Model server returned HTTP {error.code}.") from error
    except (URLError, OSError, HTTPException) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, ssl.SSLCertVerificationError):
            host = urlsplit(url).hostname
            raise DownloadCertificateError(
                f"Could not verify the TLS certificate for {host}: {reason.verify_message}."
            ) from error
        raise DownloadInterrupted("Could not connect to the model server.") from error


def _remote_files(env, endpoint, source) -> tuple[str, dict[str, _RemoteFile]]:
    try:
        with _request(env, f"{endpoint}/api/models/{quote(source)}/revision/main?blobs=true") as response:
            data = json.load(response)
    except (OSError, HTTPException) as error:
        raise DownloadInterrupted("Model metadata download was interrupted.") from error
    try:
        revision = data["sha"]
        if not _REVISION.fullmatch(revision):
            raise ValueError
        files = {}
        for item in data["siblings"]:
            path = item["rfilename"]
            size = item["size"]
            if size <= 0 or not _FILENAME.fullmatch(path) or path in {".", ".."}:
                continue
            lfs = item.get("lfs")
            checksum = lfs["sha256"] if lfs else item["blobId"]
            if re.fullmatch(r"[a-f0-9]{64}" if lfs else r"[a-f0-9]{40}", checksum):
                files[path] = _RemoteFile(size, checksum)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("The model server returned invalid file metadata.") from error
    return revision, files


def _same_size(path: Path, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size
    except OSError:
        return False


def _verified(path: Path, asset: _RemoteFile) -> bool:
    if not _same_size(path, asset.size):
        return False
    with path.open("rb") as stream:
        if len(asset.checksum) == 64:
            digest = hashlib.file_digest(stream, "sha256")
        else:
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {asset.size}\0".encode("ascii"))
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    return digest.hexdigest() == asset.checksum


def _download(env, url, asset: _RemoteFile, target: Path, update):
    if _verified(target, asset):
        update(asset.size)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{asset.checksum}.part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset >= asset.size:
        if _verified(partial, asset):
            partial.replace(target)
            update(asset.size)
            return
        partial.unlink()
        offset = 0

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with _request(env, url, **headers) as response:
        if response.status == 200:
            offset = 0
        elif response.status == 206 and offset:
            expected_range = f"bytes {offset}-{asset.size - 1}/{asset.size}"
            if response.headers.get("Content-Range") != expected_range:
                raise DownloadInterrupted("The model server returned an unexpected byte range.")
        else:
            raise RuntimeError(f"Unexpected HTTP {response.status} downloading a model.")
        update(offset)
        with partial.open("ab" if offset else "wb") as stream:
            while True:
                try:
                    chunk = response.read1(_CHUNK_SIZE)
                except (HTTPException, OSError) as error:
                    raise DownloadInterrupted("Model download was interrupted.") from error
                if not chunk:
                    break
                if offset + len(chunk) > asset.size:
                    raise RuntimeError("The model download exceeded its advertised size.")
                stream.write(chunk)
                offset += len(chunk)
                update(offset)

    if offset < asset.size:
        raise DownloadInterrupted("Model download ended before all bytes arrived.")
    if not _verified(partial, asset):
        partial.unlink()
        raise RuntimeError("The model download failed its checksum verification.")
    partial.replace(target)


def _install(root: Path, env: dict[str, str], model: _Model, progress: Progress) -> None:
    endpoint = (env.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    try:
        address = urlsplit(endpoint)
        if (address.scheme not in {"https", "http"} or not address.hostname or address.username is not None
                or address.query or address.fragment or any(char.isspace() for char in endpoint)):
            raise ValueError
        address.port  # Validate an explicitly configured port.
    except ValueError:
        raise RuntimeError("The configured model server address is invalid.") from None
    revision, remote = _remote_files(env, endpoint, model.source)
    names = list(model.files.required)
    for alternatives in model.files.alternatives:
        choice = next((choice for choice in alternatives if all(name in remote for name in choice)), None)
        if choice is None:
            raise RuntimeError("The selected model repository lacks files required by this loader.")
        names.extend(choice)
    names.extend(name for name in model.files.optional if name in remote and name not in names)
    if not all(name in remote for name in names):
        raise RuntimeError("The selected model repository lacks files required by this loader.")

    cache = _cache_repo(_hub_caches(root, env)[0], model.source)
    directory = model.target if model.target is not None else cache / "snapshots" / revision
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / _PENDING_NAME
    pending.touch()
    url = f"{endpoint}/{quote(model.source)}/resolve/{revision}"
    indexes = [name for name in names if name.endswith(".index.json")]
    completed = 0

    def transfer(name, total):
        nonlocal completed
        _download(env, f"{url}/{quote(name)}", remote[name], directory / name,
                  lambda amount: progress(completed + amount, total))
        completed += remote[name].size

    for name in indexes:
        transfer(name, None)
        shards = _index_shards(directory / name)
        if not shards or not all(shard in remote for shard in shards):
            raise RuntimeError("The selected model repository has an incomplete weight index.")
        names.extend(shard for shard in shards if shard not in names)

    total = sum(remote[name].size for name in names)
    for name in names:
        if name not in indexes:
            transfer(name, total)

    if model.target is None:
        reference = cache / "refs" / "main"
        reference.parent.mkdir(parents=True, exist_ok=True)
        temporary = reference.with_name("main.setup-part")
        temporary.write_text(revision, encoding="ascii")
        temporary.replace(reference)
    pending.unlink()
    if _ready_path(root, env, model) is None:
        raise RuntimeError("Downloaded model files failed the local completeness check.")
