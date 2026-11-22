from contextlib import contextmanager
from io import StringIO
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from core.setup_types import DownloadInterrupted


def test_pytorch_inventory_keeps_matching_cuda_wheels(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "torch", "2.11.0+cu130", "torch")
    _wheel(tmp_path, "torchvision", "0.26.0+cu130", "torchvision", "torch ==2.11.0")

    assert _items(setup, _root(tmp_path, "torch==2.11.0\ntorchvision==0.26.0"))["main:torch"].is_ready()


def test_pytorch_inventory_rejects_mixed_cpu_and_cuda_wheels(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "torch", "2.11.0+cu130", "torch")
    _wheel(tmp_path, "torchvision", "0.26.0+cpu", "torchvision", "torch ==2.11.0")

    assert not _items(setup, _root(tmp_path, "torch==2.11.0\ntorchvision==0.26.0"))["main:torch"].is_ready()


def test_live_transformers_does_not_satisfy_main_transformers(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("LIVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("MANGALENS_LIVE_PYTHON", raising=False)
    root = _root(tmp_path, "transformers==4.57.1", "transformers==5.23.0")
    _wheel(tmp_path, "transformers", "5.23.0", "transformers")
    _wheel(root / "models" / "live-runtime", "transformers", "5.23.0", "transformers")

    items = _items(setup, root)

    assert not items["main:transformers"].is_ready()
    assert items["live:transformers"].is_ready()


def test_metadata_without_recorded_module_does_not_unlock_startup(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "transitional-test-wheel", "1.0", "transitional_test_wheel")
    (tmp_path / "transitional_test_wheel" / "__init__.py").unlink()

    root = _root(tmp_path, "transitional-test-wheel==1.0")
    assert not _items(setup, root)["main:transitional-test-wheel"].is_ready()


def test_installer_only_passes_selected_package_to_pip(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    commands = []

    def pip(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="success")

    _mock_pip(monkeypatch, setup, pip)
    root = _root(tmp_path, "sample-missing==3.0\nother-missing==7.0")
    _items(setup, root)["main:sample-missing"].install(lambda *_: None)

    assert len(commands) == 1
    assert commands[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert commands[0][-1] == "sample-missing==3.0"
    assert "other-missing==7.0" not in commands[0]


def test_resolving_other_packages_preserves_installed_cuda_pins(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "torch", "2.10.0+cu128", "torch")
    _wheel(tmp_path, "torchvision", "0.25.0+cu128", "torchvision", "torch ==2.10.0")
    seen_constraints = []

    def pip(command, **kwargs):
        path = Path(command[command.index("--constraint") + 1])
        seen_constraints.extend(path.read_text(encoding="utf-8").splitlines())
        return subprocess.CompletedProcess(command, 0, stdout="success")

    _mock_pip(monkeypatch, setup, pip)
    root = _root(tmp_path, "torch==2.11.0\ntorchvision==0.26.0\nnumpy==2.3.5\npaddlex[ocr-core]==3.7.1")
    _items(setup, root)["main:paddlex"].install(lambda *_: None)

    assert "torch==2.10.0+cu128" in seen_constraints
    assert "torchvision==0.25.0+cu128" in seen_constraints
    assert "numpy==2.3.5" in seen_constraints
    assert "paddlex==3.7.1" in seen_constraints


def test_main_scan_rejects_multiple_opencv_wheels_in_the_same_directory(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "opencv-contrib-python", "4.10.0.84", "cv2")
    _wheel(tmp_path, "opencv-python", "4.10.0.84", "cv2")
    root = _root(tmp_path, "opencv-contrib-python==4.10.0.84")

    assert not _items(setup, root)["main:opencv-contrib-python"].is_ready()


def test_opencv_reinstalls_the_required_wheel_locally_when_shared_copy_conflicts(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    shared = tmp_path / "shared"
    local = tmp_path / "local"
    local.mkdir()
    _wheel(shared, "opencv-python-headless", "4.13.0.92", "cv2")
    _wheel(shared, "opencv-contrib-python", "4.10.0.84", "cv2")
    monkeypatch.syspath_prepend(str(shared))
    monkeypatch.syspath_prepend(str(local))
    monkeypatch.setattr(setup.sysconfig, "get_path", lambda name: str(local))
    root = _root(tmp_path, "opencv-contrib-python==4.10.0.84")

    def pip(command, **kwargs):
        if "--force-reinstall" in command and "--no-deps" in command:
            _wheel(local, "opencv-contrib-python", "4.10.0.84", "cv2")
        return subprocess.CompletedProcess(command, 0, stdout="success")

    _mock_pip(monkeypatch, setup, pip)
    item = _items(setup, root)["main:opencv-contrib-python"]
    assert not item.is_ready()
    item.install(lambda *_: None)

    assert item.is_ready()
    assert (shared / "opencv_python_headless-4.13.0.92.dist-info").is_dir()


def test_matching_paddle_cuda_wheel_satisfies_both_modes(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.delenv("LIVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("MANGALENS_LIVE_PYTHON", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "paddlepaddle-gpu", "3.3.1", "paddle")
    root = _root(tmp_path, "paddlepaddle==3.3.1", "paddlepaddle==3.3.1")

    items = _items(setup, root)

    assert items["main:paddlepaddle"].is_ready()
    assert items["live:paddlepaddle"].is_ready()


def test_pip_success_without_staged_payload_preserves_previous_live_runtime(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.delenv("LIVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("MANGALENS_LIVE_PYTHON", raising=False)
    root = _root(tmp_path, "", "transformers==5.23.0")
    runtime = root / "models" / "live-runtime"
    _wheel(runtime, "transformers", "4.57.1", "transformers")

    def pip(command, **kwargs):
        staging = Path(command[command.index("--target") + 1])
        _wheel(staging, "transformers", "5.23.0", "transformers")
        (staging / "transformers" / "__init__.py").unlink()
        return subprocess.CompletedProcess(command, 0, stdout="success")

    _mock_pip(monkeypatch, setup, pip)
    with pytest.raises(RuntimeError, match="transformers"):
        _items(setup, root)["live:transformers"].install(lambda *_: None)

    assert (runtime / "transformers-4.57.1.dist-info").is_dir()
    assert not (runtime / "transformers-5.23.0.dist-info").exists()


def test_live_install_preserves_other_packages_and_replaces_stale_metadata(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.delenv("LIVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("MANGALENS_LIVE_PYTHON", raising=False)
    root = _root(tmp_path, "", "transformers==5.23.0\ncustom-live-wheel==3.0")
    runtime = root / "models" / "live-runtime"
    _wheel(runtime, "transformers", "4.57.1", "transformers")
    _wheel(runtime, "custom-live-wheel", "3.0", "custom_live_wheel")
    (runtime / "transformers" / "old.py").touch()
    path = runtime / "transformers-4.57.1.dist-info" / "RECORD"
    with path.open("a", encoding="utf-8") as record:
        record.write("transformers/old.py,,\n")

    def pip(command, **kwargs):
        staging = Path(command[command.index("--target") + 1])
        assert command[-1] == "transformers==5.23.0"
        assert "custom-live-wheel==3.0" not in command
        _wheel(staging, "transformers", "5.23.0", "transformers")
        return subprocess.CompletedProcess(command, 0, stdout="success")

    _mock_pip(monkeypatch, setup, pip)
    items = _items(setup, root)
    items["live:transformers"].install(lambda *_: None)

    assert items["live:transformers"].is_ready()
    assert items["live:custom-live-wheel"].is_ready()
    assert not (runtime / "transformers" / "old.py").exists()
    assert not (runtime / "transformers-4.57.1.dist-info").exists()


def test_live_scan_uses_configured_dotenv_overlay(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.delenv("LIVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("MANGALENS_LIVE_PYTHON", raising=False)
    root = _root(tmp_path, "", "separate-live-wheel==2.0")
    (root / ".env").write_text("LIVE_PYTHONPATH=runtime-override\n", encoding="utf-8")
    _wheel(root / "runtime-override", "separate-live-wheel", "2.0", "separate_live_wheel")

    assert _items(setup, root)["live:separate-live-wheel"].is_ready()
    assert not (root / "models").exists()


def test_live_worker_checks_the_alternate_interpreters_torch_pair(tmp_path, monkeypatch):
    from core import setup_libraries as setup

    monkeypatch.delenv("LIVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("MANGALENS_LIVE_PYTHON", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    _wheel(tmp_path, "torch", "2.11.0+cpu", "torch")
    _wheel(tmp_path, "torchvision", "0.26.0+cpu", "torchvision", "torch ==2.11.0")
    root = _root(tmp_path, "torch==2.11.0\ntorchvision==0.26.0", "transformers==5.23.0")
    (root / ".env").write_text("MANGALENS_LIVE_PYTHON=worker/python.exe\n", encoding="utf-8")
    external = tmp_path / "external-site-packages"
    _wheel(external, "transformers", "5.23.0", "transformers")

    def inspect(command, **kwargs):
        import json

        assert command[0] == str(root / "worker" / "python.exe")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps([str(external)]))

    monkeypatch.setattr(setup.subprocess, "run", inspect)
    items = _items(setup, root)

    assert items["main:torch"].is_ready()
    assert not items["live:torch"].is_ready()
    assert items["live:transformers"].is_ready()


@pytest.mark.parametrize(
    ("message", "error_type"),
    [
        ("ERROR: Could not install packages due to an OSError: HTTPS read timed out", DownloadInterrupted),
        ("ERROR: No matching distribution found for sample-missing==3.0", RuntimeError),
        ("ERROR: [WinError 5] Access is denied", RuntimeError),
    ],
)
def test_pip_failure_keeps_retriable_network_errors_distinct(tmp_path, monkeypatch, message, error_type):
    from core import setup_libraries as setup

    def pip(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout=message)

    _mock_pip(monkeypatch, setup, pip)
    item = _items(setup, _root(tmp_path, "sample-missing==3.0"))["main:sample-missing"]

    with pytest.raises(error_type, match="sample-missing") as failure:
        item.install(lambda *_: None)

    assert type(failure.value) is error_type
    assert message in str(failure.value)


def _mock_pip(monkeypatch, setup, pip):
    @contextmanager
    def popen(command, **kwargs):
        result = pip(command, **kwargs)
        with StringIO(result.stdout) as stdout:
            yield SimpleNamespace(stdout=stdout, wait=lambda: result.returncode)

    monkeypatch.setattr(setup.subprocess, "Popen", popen)


def _root(tmp_path, main, live=""):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "requirements.txt").write_text(main, encoding="utf-8")
    (root / "requirements-live.txt").write_text(live, encoding="utf-8")
    (root / "requirements-nodeps.txt").touch()
    return root


def _items(setup, root):
    return {item.key: item for item in setup.library_items(root)}


def _wheel(directory: Path, name, version, module, requirement=None):
    package = directory / module
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()
    dist = directory / f"{name.replace('-', '_')}-{version}.dist-info"
    dist.mkdir()
    dependencies = f"Requires-Dist: {requirement}\n" if requirement else ""
    (dist / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n{dependencies}", encoding="utf-8")
    (dist / "RECORD").write_text(f"{module}/__init__.py,,\n", encoding="utf-8")
