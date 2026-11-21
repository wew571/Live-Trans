"""Offline model inventory and explicit, verified downloads for MangaLens."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import is_zipfile

from core.setup_environment import read_setup_environment
from core.setup_types import DownloadInterrupted, InstallationItem, Progress


_MITSUA = "Mitsua/elan-mt-bt-ja-en"
_VINAI = "vinai/vinai-translate-en2vi-v2"
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
        _Model("translate-ja-en", "Japanese → English (Mitsua ElanMT)", _MITSUA, _PROFILES["marian"]),
        _Model("live-translate-en-vi", "English → Vietnamese (VinAI Translate v2)", _VINAI, _PROFILES["mbart"]),
    ]
    for role in ("det", "rec"):
        name = f"PP-OCRv6_small_{role}_safetensors"
        models.append(_Model(
            f"live-ocr-{role}", f"Live OCR ({name})", f"PaddlePaddle/{name}", _PROFILES["live-ocr"],
            root / "models" / "live-ocr" / name,
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


def model_items(root: Path) -> list[InstallationItem]:
    """Build distinct model items without making network or filesystem changes."""
    root = root.resolve()
    def check(model):
        return _ready_path(root, read_setup_environment(root), model) is not None

    def install(model, progress):
        env = read_setup_environment(root)
        if _ready_path(root, env, model) is None:
            _install(root, env, model, progress)

    return [
        InstallationItem(
            model.key, model.label, "model",
            lambda model=model: check(model), lambda progress, model=model: install(model, progress),
        )
        for model in _model_specs(root).values()
    ]


def _request(env, url, **headers):
    request = Request(url, headers={"Accept-Encoding": "identity", **headers})
    if token := env.get("HF_TOKEN"):
        if not token.isascii() or not token.isprintable() or " " in token:
            raise RuntimeError("The configured Hugging Face token has an invalid format.")
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return urlopen(request, timeout=30)
    except HTTPError as error:
        error.close()
        if error.code in {408, 425, 429, 500, 502, 503, 504}:
            raise DownloadInterrupted(f"Model server temporarily returned HTTP {error.code}.") from error
        raise RuntimeError(f"Model server returned HTTP {error.code}.") from error
    except (URLError, OSError, HTTPException) as error:
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
