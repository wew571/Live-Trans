import builtins
import importlib
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_importing_setup_entrypoint_works_without_site_packages():
    script = dedent("""
        from pathlib import Path
        import main
        from core.setup_libraries import library_items
        from core.setup_models import model_items
        from frontend.setup_window import _scan

        root = Path.cwd()
        items = [*library_items(root), *model_items(root)]
        result = _scan(items)
        assert any(item.category == 'library' for item in result.missing)
        assert not result.errors, result.errors
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ready", [False, True])
def test_main_only_loads_control_panel_when_setup_succeeds(monkeypatch, ready):
    main = importlib.import_module("main")
    events = []
    modules = {
        "core.setup_libraries": SimpleNamespace(
            library_items=lambda path: ["library"], opencv_install_conflicts=lambda: [],
        ),
        "core.setup_models": SimpleNamespace(model_items=lambda path: ["model"]),
        "frontend.setup_window": SimpleNamespace(
            run_setup=lambda items: events.append(list(items)) or ready,
        ),
    }
    original_import = builtins.__import__

    def intercept(name, *args, **kwargs):
        if name == "frontend.ui":
            pytest.fail("The setup process loaded the application UI")
        if name in modules:
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercept)
    monkeypatch.setattr(main, "_launch_application", lambda root: events.append("app") or 0)
    assert main.main() == 0
    expected = [["library", "model"], "app"] if ready else [["library", "model"]]
    assert events == expected


def test_main_uses_project_python_when_the_shared_python_has_opencv_conflicts(tmp_path, monkeypatch):
    import main
    from core import setup_libraries
    from frontend import setup_window

    shared = tmp_path / "shared-python"
    shared.mkdir()
    metadata = shared / "opencv_python_headless-4.13.0.92.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: opencv-python-headless\nVersion: 4.13.0.92\n", encoding="utf-8")
    project = tmp_path / "project"
    environment = project / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    (environment / "pyvenv.cfg").touch()
    monkeypatch.setattr(main, "__file__", str(project / "main.py"))
    monkeypatch.setattr(setup_libraries.sysconfig, "get_path", lambda name: str(shared))
    monkeypatch.setattr(setup_window, "run_setup", lambda items: pytest.fail("Setup ran in shared Python"))
    calls = []
    monkeypatch.setattr(main.subprocess, "call", lambda command, cwd: calls.append((command, cwd)) or 17)

    assert main.main() == 17
    assert calls == [([str(python), str(project / "main.py")], project)]
    assert (metadata / "METADATA").is_file()


def test_launcher_runs_with_a_fresh_python_and_project_working_directory(tmp_path):
    import main

    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "__init__.py").touch()
    (frontend / "ui.py").write_text(
        "class ControlPanel:\n"
        "    def mainloop(self):\n"
        "        from pathlib import Path\n"
        "        Path('launched.txt').write_text('running', encoding='utf-8')\n",
        encoding="utf-8",
    )
    assert main._launch_application(tmp_path) == 0
    assert (tmp_path / "launched.txt").read_text(encoding="utf-8") == "running"



def test_dashboard_is_visible_when_its_first_mainloop_starts():
    script = dedent("""
        import keyboard
        import win32con
        import win32gui

        from frontend.ui import ControlPanel

        keyboard.add_hotkey = lambda *args: None
        panel = ControlPanel()
        state = []

        def inspect_startup():
            try:
                hwnd = win32gui.GetAncestor(panel.winfo_id(), win32con.GA_ROOT)
                state.extend((panel.state(), bool(win32gui.IsWindowVisible(hwnd))))
            finally:
                panel.quit_all()

        panel.after(250, inspect_startup)
        panel.mainloop()
        assert state == ["normal", True], state
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True,
        text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
