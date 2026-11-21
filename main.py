from pathlib import Path
import subprocess
import sys
import venv


def _project_setup_python(root: Path) -> Path | None:
    """Find a project environment that can install OpenCV outside the current site-packages."""
    managed = root / "models" / "setup-python"
    current = Path(sys.prefix).resolve()
    if current == managed.resolve():
        return None

    executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    project = root / ".venv"
    project_python = project / executable
    if current != project.resolve() and (project / "pyvenv.cfg").is_file() and project_python.is_file():
        return project_python

    python = managed / executable
    if not (managed / "pyvenv.cfg").is_file() or not python.is_file():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(managed)
    return python


def _launch_application(root: Path) -> int:
    # A fresh interpreter processes .pth files added during setup (for example pywin32).
    return subprocess.call(
        [sys.executable, "-c", "from frontend.ui import ControlPanel; ControlPanel().mainloop()"],
        cwd=root,
    )


def main() -> int:
    from core.setup_libraries import library_items, opencv_install_conflicts

    root = Path(__file__).resolve().parent
    if opencv_install_conflicts() and (python := _project_setup_python(root)) is not None:
        return subprocess.call([str(python), str(root / "main.py")], cwd=root)

    from core.setup_models import model_items
    from frontend.setup_window import run_setup

    if not run_setup([*library_items(root), *model_items(root)]):
        return 0
    return _launch_application(root)


if __name__ == "__main__":
    raise SystemExit(main())
