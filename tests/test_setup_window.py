import gc
import importlib.util
import tkinter as tk
from threading import Event
from time import monotonic
from tkinter import ttk

import pytest


@pytest.fixture
def setup_window():
    assert importlib.util.find_spec("frontend.setup_window") is not None
    from frontend import setup_window

    return setup_window


@pytest.fixture
def drive_window(setup_window, monkeypatch):
    # Dispose of previous Tk test windows on the GUI thread before a worker can trigger GC.
    gc.collect()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    monkeypatch.setattr(setup_window.tk, "Tk", lambda: root)

    def drive(items, steps=()):
        remaining = iter(steps)
        step = next(remaining, None)
        deadline = monotonic() + 6
        errors = []
        scheduled = None

        def tick():
            nonlocal step, scheduled
            try:
                if step and step(root):
                    step = next(remaining, None)
                if monotonic() > deadline:
                    raise AssertionError("Setup did not reach the expected state")
                scheduled = root.after(10, tick)
            except Exception as exc:
                errors.append(exc)
                root.quit()

        def cancel_tick(event):
            if event.widget is root:
                root.after_cancel(scheduled)

        scheduled = root.after(10, tick)
        root.bind("<Destroy>", cancel_tick)
        result = setup_window.run_setup(items)
        if errors:
            raise errors[0]
        assert step is None
        return result

    return drive


def widgets(root):
    for child in root.winfo_children():
        yield child
        yield from widgets(child)


def find(root, widget_type, label):
    return next(
        (child for child in widgets(root) if isinstance(child, widget_type)
         and child.cget("text") == label and child.winfo_ismapped()),
        None,
    )


def press(label, widget_type=ttk.Button):
    def step(root):
        button = next(
            (child for child in widgets(root) if isinstance(child, widget_type)
             and child.cget("text") == label and child.winfo_ismapped() and child.instate(["!disabled"])),
            None,
        )
        if button is not None:
            button.invoke()
            return True
        return False

    return step


def item(key, category, ready, install):
    from core.setup_types import InstallationItem

    return InstallationItem(key, key, category, ready, install)


def close(root):
    root.tk.call(root.protocol("WM_DELETE_WINDOW"))
    return True


def test_setup_shows_live_trans_branding(drive_window):
    def check(root):
        assert root.title() == "Thiết lập Live Trans"
        if find(root, ttk.Label, "Thiết lập Live Trans") is None:
            return False
        return close(root)

    missing = item("OCR", "model", lambda: False, lambda progress: None)
    assert not drive_window([missing], [check])


def test_complete_launch_scans_all_items_and_does_not_install(drive_window):
    checked = []
    installed = []
    items = [
        item(key, category, lambda key=key: checked.append(key) or True,
             lambda progress: installed.append(True))
        for key, category in [("numpy", "library"), ("OCR", "model")]
    ]

    assert drive_window(items) is True
    assert checked == ["numpy", "OCR"]
    assert installed == []


def test_startup_can_continue_after_required_install_without_downloading_addons(drive_window):
    from dataclasses import replace

    installed = set()
    base = item("OCR Small", "model", lambda: "Small" in installed, lambda _: installed.add("Small"))
    addon = replace(item("OCR Medium", "model", lambda: False, lambda _: installed.add("Medium")),
                    optional=True)

    def check_groups(root):
        if not (find(root, ttk.Checkbutton, "OCR Medium") and find(root, ttk.Label, "Mô hình bổ sung")):
            return False
        assert not press("Tiếp tục")(root)
        return True

    assert drive_window([base, addon], [check_groups, press("OCR Small", ttk.Checkbutton),
                                       press("Cài mô hình đã chọn"), press("Tiếp tục")])
    assert installed == {"Small"}


def test_only_missing_addons_do_not_reopen_setup_on_launch(drive_window):
    from dataclasses import replace

    optional = replace(item("NLLB", "model", lambda: False, lambda _: pytest.fail("unexpected download")),
                       optional=True)
    assert drive_window([optional]) is True


def test_confirmed_model_install_uses_modal_and_restores_parent_grab(setup_window):
    root = tk.Tk()
    try:
        settings = tk.Toplevel(root)
        settings.grab_set()
        installed = []
        nllb = item("NLLB", "model", lambda: bool(installed), lambda _: installed.append("NLLB"))
        root.after(5000, lambda: [close(window) for window in settings.winfo_children()])
        assert setup_window.run_setup([nllb], parent=settings, install_now=True)
        assert installed == ["NLLB"]
        assert root.grab_current() == settings
    finally:
        root.destroy()


def test_install_stays_in_the_selected_group_until_user_starts_the_other(drive_window):
    installed = set()
    items = [
        item(key, category, lambda key=key: key in installed,
             lambda progress, key=key: installed.add(key))
        for key, category in [("numpy", "library"), ("Pillow", "library"), ("OCR", "model")]
    ]

    def model_list_after_library_install(root):
        ready = find(root, ttk.Label, "Chọn các mục còn thiếu để cài đặt.")
        if ready and find(root, ttk.Checkbutton, "Pillow") and installed == {"numpy"}:
            assert find(root, ttk.Checkbutton, "numpy") is None
            assert find(root, ttk.Checkbutton, "OCR") is not None
            return close(root)
        return False

    assert drive_window(items, [press("numpy", ttk.Checkbutton), press("Cài thư viện đã chọn"),
                                model_list_after_library_install]) is False
    assert installed == {"numpy"}


def test_library_output_is_visible_during_install_and_remains_afterwards(drive_window, tmp_path):
    from dataclasses import replace

    from core.setup_libraries import library_items

    (tmp_path / "requirements.txt").write_text("sample-terminal-test==1.0", encoding="utf-8")
    (tmp_path / "requirements-nodeps.txt").touch()
    (tmp_path / "requirements-live.txt").touch()
    (tmp_path / "pip.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "from time import monotonic, sleep\n"
        "print('Downloading sample library')\n"
        "print('Sample warning', file=sys.stderr)\n"
        "deadline = monotonic() + 3\n"
        "while not Path('release').exists() and monotonic() < deadline:\n"
        "    sleep(0.01)\n"
        "Path('finished').touch()\n",
        encoding="utf-8",
    )
    library, = library_items(tmp_path)
    finished = tmp_path / "finished"
    library = replace(library, is_ready=finished.exists)
    model = item("OCR", "model", lambda: False, lambda progress: None)

    def see_live_output(root):
        assert not finished.exists(), "The installer exited before its output became visible"
        terminal = next((child for child in widgets(root) if isinstance(child, tk.Text)
                         and child.winfo_ismapped()), None)
        if terminal is None:
            return False
        output = terminal.get("1.0", "end")
        if "Downloading sample library" not in output or "Sample warning" not in output:
            return False
        assert terminal.cget("state") == "disabled"
        (tmp_path / "release").touch()
        return True

    def see_saved_output(root):
        if not find(root, ttk.Label, "Chọn các mục còn thiếu để cài đặt."):
            return False
        terminal = next(child for child in widgets(root) if isinstance(child, tk.Text))
        assert terminal.winfo_ismapped()
        assert "Downloading sample library" in terminal.get("1.0", "end")
        return close(root)

    assert not drive_window([library, model], [press(library.label, ttk.Checkbutton),
                                              press("Cài thư viện đã chọn"),
                                              see_live_output, see_saved_output])


def test_try_now_preserves_model_progress_and_pending_install_order(drive_window):
    from core.setup_types import DownloadInterrupted

    calls = []
    installed = set()

    def install_ocr(progress):
        calls.append("OCR")
        if calls.count("OCR") == 1:
            progress(25, 100)
            raise DownloadInterrupted("connection reset")
        installed.add("OCR")

    def retry_interrupted_model(root):
        if not find(root, ttk.Button, "Thử ngay"):
            return False
        progress = next(child for child in widgets(root) if isinstance(child, ttk.Progressbar))
        assert float(progress.cget("value")) == 25.0
        assert calls == ["OCR"]
        return press("Thử ngay")(root)

    def install_translation(progress):
        calls.append("Dịch")
        installed.add("Dịch")

    items = [item("OCR", "model", lambda: "OCR" in installed, install_ocr),
             item("Dịch", "model", lambda: "Dịch" in installed, install_translation)]
    assert drive_window(items, [press("OCR", ttk.Checkbutton), press("Dịch", ttk.Checkbutton),
                                press("Cài mô hình đã chọn"), retry_interrupted_model]) is True
    assert calls == ["OCR", "OCR", "Dịch"]


def test_select_all_models_automatically_retries_when_download_recovers(drive_window, setup_window, monkeypatch):
    from core.setup_types import DownloadInterrupted

    monkeypatch.setattr(setup_window, "_BACKOFF_SECONDS", (0.05,))
    completed = set()
    attempts = []

    def install(name):
        attempts.append(name)
        if attempts == ["OCR"]:
            raise DownloadInterrupted("connection reset")
        completed.add(name)

    items = [
        item(name, "model", lambda name=name: name in completed, lambda progress, name=name: install(name))
        for name in ("OCR", "Dịch")
    ]

    assert drive_window(items, [press("Chọn tất cả"), press("Cài mô hình đã chọn")]) is True
    assert attempts == ["OCR", "OCR", "Dịch"]


def test_certificate_failure_shows_https_hint_and_manual_retry(drive_window):
    from core.setup_types import DownloadCertificateError

    def install(progress):
        raise DownloadCertificateError("Could not verify the TLS certificate for github.com.")

    def check(root):
        if not find(root, ttk.Button, "Thử lại"):
            return False
        details = [child.cget("text") for child in widgets(root) if isinstance(child, ttk.Label)]
        assert any("Không thể xác minh chứng chỉ HTTPS" in text and "github.com" in text for text in details)
        assert find(root, ttk.Button, "Thử ngay") is None
        return close(root)

    missing = item("Qwen CUDA", "model", lambda: False, install)
    assert not drive_window([missing], [press(missing.label, ttk.Checkbutton), press("Cài mô hình đã chọn"), check])


def test_installer_return_without_ready_assets_requires_manual_retry(drive_window):
    attempts = []

    def install(progress):
        attempts.append(True)

    items = [item("OCR", "model", lambda: len(attempts) == 2, install)]
    assert drive_window(items, [press("OCR", ttk.Checkbutton), press("Cài mô hình đã chọn"),
                                press("Thử lại")]) is True
    assert len(attempts) == 2


def test_close_waits_for_active_operation_and_skips_later_items(drive_window):
    started = Event()
    release = Event()
    calls = []

    def install(progress):
        started.set()
        assert release.wait(5)
        calls.append("OCR")

    def close_after_start(root):
        if not started.is_set():
            return False
        close(root)
        root.after(50, release.set)
        return True

    items = [item("OCR", "model", lambda: "OCR" in calls, install),
             item("Dịch", "model", lambda: False, lambda progress: calls.append("Dịch"))]
    assert drive_window(items, [press("OCR", ttk.Checkbutton), press("Dịch", ttk.Checkbutton),
                                press("Cài mô hình đã chọn"), close_after_start]) is False
    assert calls == ["OCR"]
