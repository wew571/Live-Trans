"""Inventory and install the Python dependencies before importing the application."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from importlib import metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory, mkdtemp

from core.config import APP_NAME
from core.setup_environment import read_setup_environment
from core.setup_types import DownloadInterrupted, InstallationItem, InstallLog, Progress


_PIN = re.compile(r"([\w.-]+)(\[[\w,-]+\])?==([\w.+-]+)")
_OPENCV = "opencv-contrib-python"
_OTHER_OPENCV = ("opencv-python", "opencv-python-headless", "opencv-contrib-python-headless")
_PADDLE = {"paddleocr", "paddlex", _OPENCV}
_PADDLE_CPU = "paddlepaddle"
_PADDLE_GPU = "paddlepaddle-gpu"
_TORCH = {"torch", "torchvision"}
_OPTIONAL_MAIN = {"pywebview"}
_NETWORK_FAILURE = re.compile(
    r"(?:time(?:d)? ?out|readtimeout|connecttimeout|connection (?:reset|aborted|refused|broken)|"
    r"remote end closed|incomplete ?read|broken pipe|network is unreachable|"
    r"failed to establish a new connection|getaddrinfo failed|temporary failure in name resolution|"
    r"(?:429|500|502|503|504) (?:client error|server error|error response|too many requests|"
    r"bad gateway|service unavailable|gateway timeout|internal server error))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Pin:
    name: str
    version: str
    extras: str = ""

    @property
    def requirement(self) -> str:
        return f"{self.name}{self.extras}=={self.version}"


def _read_pins(path: Path) -> list[_Pin]:
    pins = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        requirement = line.partition("#")[0].strip()
        if not requirement:
            continue
        match = _PIN.fullmatch(requirement)
        if not match:
            raise ValueError(f"Expected a pinned package in {path.name}: {requirement}")
        package, extras, version = match.groups()
        name = re.sub(r"[-_.]+", "-", package.lower())
        pins.append(_Pin(name, version, extras or ""))
    return pins


def _distribution(name: str, paths=None):
    return next(metadata.Distribution.discover(name=name, path=sys.path if paths is None else paths), None)


def _has_payload(distribution: metadata.Distribution) -> bool:
    """A stale dist-info directory does not make a partially removed wheel ready."""
    records = distribution.files or ()
    source = [record for record in records if record.name == "__init__.py"]
    extensions = [record for record in records if record.suffix in (".pyd", ".so")]
    if not source and not extensions:
        source = [record for record in records if record.suffix == ".py"]
    if not source and not extensions:
        # Native-only wheels such as nvidia-cublas-cu12 contain only libraries.
        extensions = [record for record in records if record.suffix == ".dll"]
    if not source and not extensions:
        return False
    first_sources = sorted(source, key=lambda record: len(record.parts))[:1]
    return all(distribution.locate_file(record).is_file() for record in first_sources + extensions)


def _pinned(pin: _Pin, paths=None) -> bool:
    distribution = _distribution(pin.name, paths)
    if pin.name == _PADDLE_CPU:
        providers = (distribution, _distribution(_PADDLE_GPU, paths))
        active = [wheel for wheel in providers if wheel is not None and _provides("paddle", wheel, paths)]
        distribution = active[0] if len(active) == 1 else None
    if distribution is None or not _matches(pin.version, distribution.version):
        return False
    if pin.name == _OPENCV:
        directory = str(distribution.locate_file(""))
        if any(_distribution(name, [directory]) for name in _OTHER_OPENCV):
            return False
        if not _provides("cv2", distribution, paths):
            return False
    return _has_payload(distribution)


def _provides(module, distribution, paths):
    spec = PathFinder.find_spec(module, paths)
    return bool(
        spec and spec.origin and Path(spec.origin).resolve().is_relative_to(distribution.locate_file(module).resolve())
    )


def _matches(expected: str, actual: str | None) -> bool:
    if actual is None:
        return False
    return expected.lower() == (actual if "+" in expected else actual.split("+", 1)[0]).lower()


def _torch_pair(paths=None) -> tuple[str, str] | None:
    torch = _distribution("torch", paths)
    vision = _distribution("torchvision", paths)
    if torch is None or vision is None or not torch.version or not vision.version:
        return None
    torch_release, _, torch_build = torch.version.partition("+")
    vision_build = vision.version.partition("+")[2]
    release = re.fullmatch(r"(\d+)\.(\d+)\.\d+", torch_release)
    if release is None or tuple(map(int, release.groups())) < (2, 6) or torch_build != vision_build:
        return None
    required_torch = [
        match[1]
        for requirement in vision.requires or ()
        if (match := re.fullmatch(r"torch\s*\(?\s*==\s*([\w.+-]+)\s*\)?", requirement))
    ]
    if not required_torch or not _matches(required_torch[0], torch.version):
        return None
    if not _has_payload(torch) or not _has_payload(vision):
        return None
    return torch.version, vision.version


def _pip(executable, arguments, root, progress: Progress, label, log: InstallLog | None):
    command = [executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
               "--no-color", *arguments]
    progress(0, None)
    lines = []
    try:
        with subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as process:
            for line in process.stdout:
                lines.append(line)
                if log is not None:
                    log(line.rstrip("\n"))
            returncode = process.wait()
    except OSError as error:
        raise RuntimeError(f"Could not run pip to install {label}: {error}") from error
    if returncode:
        output = "".join(lines).strip()
        detail = "\n".join(output.splitlines()[-12:]) or f"pip exited with code {returncode}"
        message = f"Could not install {label} using {executable}.\n{detail}"
        if _NETWORK_FAILURE.search(output) and not re.search(r"certificate verify failed|sslcertverificationerror", output, re.I):
            raise DownloadInterrupted(message)
        raise RuntimeError(message)


def opencv_install_conflicts() -> list[str]:
    install_dirs = {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
    return [name for name in _OTHER_OPENCV if _distribution(name, install_dirs)]


def _check_opencv_install_directory():
    conflicts = opencv_install_conflicts()
    if conflicts:
        wheels = " ".join([*conflicts, _OPENCV])
        raise RuntimeError(
            "OpenCV wheels share the same cv2 files. Remove the existing variants from this "
            f"Python environment before installing PaddleOCR/OpenCV:\n{sys.executable} -m pip uninstall {wheels}"
        )


def _install_main(root, label, selected, constraints, no_deps, progress, log=None):
    torch_selected = all(pin.name in _TORCH for pin in selected)
    ready = bool(_torch_pair()) if torch_selected else all(_pinned(pin) for pin in selected)
    if ready:
        progress(1, 1)
        return
    if any(pin.name in _PADDLE for pin in selected):
        _check_opencv_install_directory()
    if any(pin.name == _PADDLE_CPU for pin in selected) and _distribution(
        _PADDLE_GPU, {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
    ):
        raise RuntimeError(
            f"Install the matching PaddlePaddle GPU version or remove the existing GPU wheel "
            f"before installing {label}. Both wheels write to the same paddle package."
        )
    with TemporaryDirectory(prefix="mangalens-pip-") as directory:
        constraint_file = Path(directory) / "constraints.txt"
        compatible_torch = _torch_pair()
        versions = dict(zip(("torch", "torchvision"), compatible_torch or (), strict=False))
        constraint_file.write_text(
            "\n".join(f"{pin.name}=={versions.get(pin.name, pin.version)}" for pin in constraints),
            encoding="utf-8",
        )
        arguments = ["--constraint", str(constraint_file)]
        # An inherited matching wheel can still be unusable because another wheel owns cv2 there.
        # Install the OpenCV wheel locally and let the separately pinned NumPy item manage NumPy.
        if len(selected) == 1 and selected[0].name == _OPENCV:
            arguments.append("--no-deps")
            if (wheel := _distribution(_OPENCV)) is not None and _matches(selected[0].version, wheel.version):
                arguments.append("--force-reinstall")
        elif no_deps:
            arguments.append("--no-deps")
        elif any(
            pin.name in _TORCH and (wheel := _distribution(pin.name)) is not None
            and _matches(pin.version, wheel.version)
            for pin in selected
        ):
            # Reinstall the pair together when one exact-version wheel has an incompatible build.
            arguments.append("--force-reinstall")
        else:
            for pin in selected:
                wheel = _distribution(pin.name)
                if wheel is not None and _matches(pin.version, wheel.version) and not _has_payload(wheel):
                    arguments.append("--force-reinstall")
                    break
        if no_deps and any(
            (wheel := _distribution(pin.name)) is not None and _matches(pin.version, wheel.version)
            for pin in selected
        ):
            arguments.append("--force-reinstall")
        _pip(sys.executable, [*arguments, *(pin.requirement for pin in selected)], root, progress, label, log)
    progress(1, 1)


def _live_executable(root, settings):
    executable = settings.get("MANGALENS_LIVE_PYTHON")
    if not executable:
        return sys.executable
    path = Path(executable)
    if not path.is_absolute() and (path.parent != Path(".") or executable.startswith(".")):
        return str((root / path).resolve())
    return executable


def _live_paths(root, runtime):
    settings = read_setup_environment(root)
    executable = _live_executable(root, settings)
    configured_path = settings.get("LIVE_PYTHONPATH")
    overlay = str(runtime) if configured_path is None else configured_path
    if executable == sys.executable:
        return [*(str(root / entry) for entry in overlay.split(os.pathsep) if entry), *sys.path]
    pythonpath = os.pathsep.join(path for path in (overlay, str(root), settings.get("PYTHONPATH")) if path)
    return _interpreter_paths(executable, root, pythonpath)


def _interpreter_paths(executable, root, pythonpath):
    env = dict(os.environ, PYTHONPATH=pythonpath)
    try:
        result = subprocess.run(
            [executable, "-c", "import json, sys; print(json.dumps(sys.path))"],
            env=env, cwd=root, capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def _live_ready(root, runtime, pin):
    return _pinned(pin, _live_paths(root, runtime))


def _install_live(root, runtime, pin, pins, label, progress, log=None):
    if _live_ready(root, runtime, pin):
        progress(1, 1)
        return
    settings = read_setup_environment(root)
    executable = _live_executable(root, settings)
    if settings.get("LIVE_PYTHONPATH") is not None:
        raise RuntimeError(
            f"Install {pin.requirement} in the configured LIVE_PYTHONPATH, "
            f"or clear that setting to let {APP_NAME} manage its default live runtime."
        )
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".live-runtime-", dir=runtime.parent) as directory:
        work = Path(directory)
        staging = work / "packages"
        constraints = work / "constraints.txt"
        constraints.write_text("\n".join(f"{entry.name}=={entry.version}" for entry in pins), encoding="utf-8")
        _pip(
            executable, ["--target", str(staging), "--constraint", str(constraints), pin.requirement],
            root, progress, label, log,
        )
        if not _pinned(pin, [str(staging)]):
            raise RuntimeError(f"Incomplete live runtime installation: {pin.requirement}")
        merged = _merge_live_wheels(runtime, staging, work)
        _replace_runtime(runtime, merged)
    progress(1, 1)


def _merge_live_wheels(runtime, staging, work):
    if not runtime.exists():
        return staging
    merged = work / "merged"
    shutil.copytree(runtime, merged)
    installed_wheels = list(merged.glob("*.dist-info"))
    updated_names = {
        wheel.metadata["Name"].lower().replace("_", "-")
        for wheel in metadata.distributions(path=[str(staging)])
    }
    if _OPENCV in updated_names:
        updated_names.update(_OTHER_OPENCV)
    if _PADDLE_CPU in updated_names:
        updated_names.add(_PADDLE_GPU)
    for directory in installed_wheels:
        wheel = metadata.PathDistribution(directory)
        name = wheel.metadata.get("Name")
        if not name or name.lower().replace("_", "-") not in updated_names:
            continue
        for file in wheel.files or ():
            path = wheel.locate_file(file).resolve()
            if path.is_relative_to(merged.resolve()) and path.is_file():
                path.unlink()
        shutil.rmtree(directory)
    shutil.copytree(staging, merged, dirs_exist_ok=True)
    return merged


def _replace_runtime(runtime, staged):
    if not runtime.exists():
        staged.rename(runtime)
        return
    backup = Path(mkdtemp(prefix=".live-runtime-backup-", dir=runtime.parent))
    previous = backup / runtime.name
    try:
        runtime.rename(previous)
    except OSError:
        backup.rmdir()
        raise
    try:
        staged.rename(runtime)
    except OSError:
        try:
            previous.rename(runtime)
        except OSError as error:
            raise RuntimeError(f"Could not restore the live runtime. Its previous files are at {previous}.") from error
        backup.rmdir()
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _install_worker_torch(root, torch_pins, label, progress, log=None):
    executable = _live_executable(root, read_setup_environment(root))
    command = ["--force-reinstall", *(pin.requirement for pin in torch_pins)]
    _pip(executable, command, root, progress, label, log)
    progress(1, 1)


def _library_item(key, label, is_ready, install, *, optional=False):
    return InstallationItem(key, label, "library", is_ready, install, install_with_log=install, optional=optional)


def _main_library_item(root, pin, constraints, no_deps=False):
    label = f"{pin.name} ({APP_NAME})"
    return _library_item(
        f"main:{pin.name}", label, partial(_pinned, pin),
        partial(_install_main, root, label, [pin], constraints, no_deps),
        optional=pin.name in _OPTIONAL_MAIN,
    )


def optional_library_items(root: Path) -> list[InstallationItem]:
    manifest = root / "requirements.txt"
    pins = _read_pins(manifest) if manifest.is_file() else []
    return [_main_library_item(root, pin, pins) for pin in pins if pin.name in _OPTIONAL_MAIN]


def library_items(root: Path) -> list[InstallationItem]:
    """Use the same pin manifests for validation and for pip's resolver."""
    main_pins = _read_pins(root / "requirements.txt")
    no_deps_pins = _read_pins(root / "requirements-nodeps.txt")
    live_pins = _read_pins(root / "requirements-live.txt")
    items = []
    torch_pins = [pin for pin in main_pins if pin.name in _TORCH]
    if torch_pins:
        label = f"PyTorch and torchvision ({APP_NAME})"
        items.append(_library_item(
            "main:torch", label, lambda: _torch_pair() is not None,
            partial(_install_main, root, label, torch_pins, main_pins, False),
        ))
    for pin in [*main_pins, *no_deps_pins]:
        if pin.name in _TORCH:
            continue
        items.append(_main_library_item(root, pin, main_pins, pin in no_deps_pins))
    if live_pins:
        runtime = root / "models" / "live-runtime"
        if torch_pins and _live_executable(root, read_setup_environment(root)) != sys.executable:
            label = "PyTorch and torchvision (Live Translation)"
            items.append(_library_item(
                "live:torch", label, lambda: _torch_pair(_live_paths(root, runtime)) is not None,
                partial(_install_worker_torch, root, torch_pins, label),
            ))
        for pin in live_pins:
            label = f"{pin.name} (Live Translation)"
            items.append(_library_item(
                f"live:{pin.name}", label, partial(_live_ready, root, runtime, pin),
                partial(_install_live, root, runtime, pin, live_pins, label),
            ))
    return items
