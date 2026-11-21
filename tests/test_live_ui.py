from types import SimpleNamespace
from unittest.mock import Mock

import keyboard
import pytest

from frontend.ui import ControlPanel


@pytest.fixture(scope="module")
def gui():
    # Keep one Tcl/Tk root per test process for these dashboard cases.
    with pytest.MonkeyPatch.context() as patch:
        register_hotkey = Mock(return_value=None)
        patch.setattr(keyboard, "add_hotkey", register_hotkey)
        session = Mock()
        overlay = Mock()
        dashboard = ControlPanel(
            live_session_factory=lambda: session,
            live_overlay_factory=lambda parent: overlay,
        )
        try:
            dashboard.update()
            yield dashboard, session, overlay, register_hotkey
        finally:
            dashboard.quit_all()


@pytest.fixture
def panel(gui):
    dashboard, session, overlay, _ = gui
    dashboard._stop_live()
    dashboard._set_live_capture_mode("desktop")
    dashboard._set_area(None)
    session.reset_mock()
    session.poll.return_value = None
    overlay.reset_mock()
    return dashboard, session, overlay


def test_dashboard_uses_live_trans_name_and_transparent_assets(panel):
    dashboard, _, _ = panel
    assert dashboard.title() == "Live Trans"
    assert dashboard.logo_label.cget("text") == "Live Trans"
    for widget in (dashboard.logo_label, dashboard.btn_pick_window):
        im = widget.cget("image").cget("light_image")
        assert im.mode == "RGBA"
        assert im.getchannel("A").getextrema() == (0, 255)


def test_dashboard_uses_live_trans_native_window_icons(panel):
    import win32con
    import win32gui
    import win32ui

    from core.config import APP_ICON_PATH

    def bitmap_data(icon):
        _, _, _, mask, color = win32gui.GetIconInfo(icon)
        try:
            bitmap = win32ui.CreateBitmapFromHandle(color)
            return bitmap.GetInfo()["bmWidth"], bitmap.GetBitmapBits(True)
        finally:
            win32gui.DeleteObject(mask)
            win32gui.DeleteObject(color)

    dashboard, _, _ = panel
    hwnd = win32gui.GetAncestor(dashboard.winfo_id(), win32con.GA_ROOT)
    for kind in (0, 1):
        icon = win32gui.SendMessage(hwnd, win32con.WM_GETICON, kind, 0)
        assert icon
        size, pixels = bitmap_data(icon)
        expected = win32gui.LoadImage(
            0, str(APP_ICON_PATH), win32con.IMAGE_ICON, size, size, win32con.LR_LOADFROMFILE
        )
        try:
            assert (size, pixels) == bitmap_data(expected)
        finally:
            win32gui.DestroyIcon(expected)


def test_taskbar_uses_live_trans_name_and_custom_icon(panel):
    import ctypes
    import subprocess
    import sys
    from pathlib import Path

    import win32con
    import win32gui
    from win32com.propsys import propsys, pscon

    from core.config import APP_ICON_PATH

    dashboard, _, _ = panel
    hwnd = win32gui.GetAncestor(dashboard.winfo_id(), win32con.GA_ROOT)
    store = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
    app_id = store.GetValue(pscon.PKEY_AppUserModel_ID).GetValue()
    assert app_id == "LiveTrans.Desktop"
    assert store.GetValue(pscon.PKEY_AppUserModel_RelaunchIconResource).GetValue() == f"{APP_ICON_PATH},0"
    assert store.GetValue(pscon.PKEY_AppUserModel_RelaunchDisplayNameResource).GetValue() == "Live Trans"
    launcher = Path(__file__).resolve().parents[1] / "main.py"
    assert store.GetValue(pscon.PKEY_AppUserModel_RelaunchCommand).GetValue() == subprocess.list2cmdline([
        sys.executable, str(launcher)
    ])

    current = ctypes.c_wchar_p()
    assert ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(current)) == 0
    try:
        assert current.value == app_id
    finally:
        ctypes.windll.ole32.CoTaskMemFree(current)


def test_minimizing_keeps_window_on_taskbar_and_restoring_hides_floating_controls(panel):
    import win32con
    import win32gui

    dashboard, _, _ = panel
    hwnd = win32gui.GetAncestor(dashboard.winfo_id(), win32con.GA_ROOT)
    try:
        dashboard.iconify()
        dashboard.update()
        assert win32gui.IsIconic(hwnd)
        assert win32gui.IsWindowVisible(hwnd)
        assert dashboard.floating_widget.winfo_viewable()

        dashboard.deiconify()
        dashboard.update()
        assert dashboard.state() == "normal"
        assert not dashboard.floating_widget.winfo_viewable()
    finally:
        dashboard.restore_dashboard()
        dashboard.update()


def test_only_translation_hotkey_is_registered(gui, panel):
    dashboard, session, _ = panel
    _, _, _, register_hotkey = gui

    register_hotkey.assert_called_once()
    key, callback = register_hotkey.call_args.args
    assert key == "`"

    callback()
    dashboard._ui_actions.get_nowait()()
    session.start.assert_called_once_with(dashboard.fullscreen_region())

    callback()
    dashboard._ui_actions.get_nowait()()
    session.stop.assert_called_once()


def test_region_picker_can_be_cancelled_with_a_button(gui):
    from frontend.live_overlay import RegionPicker

    dashboard, _, _, _ = gui
    finished = Mock()
    picker = RegionPicker(dashboard, finished, "Select a region", "Cancel", region=(40, 40, 320, 200))
    try:
        assert not picker.window.bind("<Escape>")
        picker.window.nametowidget("cancel").invoke()
        finished.assert_called_once_with(None)
        assert not picker.window.winfo_exists()
    finally:
        if picker.window.winfo_exists():
            picker.close()


def test_scan_button_toggles_the_same_live_session_as_monitor(panel):
    dashboard, session, overlay = panel
    assert not hasattr(dashboard, "mode_picker")

    dashboard.btn_scan.invoke()

    session.start.assert_called_once_with(dashboard.fullscreen_region())
    assert dashboard.auto_switch.get() == 1
    dashboard.auto_switch.toggle()

    session.stop.assert_called_once()
    overlay.hide.assert_called()
    assert dashboard.auto_switch.get() == 0


def test_recording_translates_selected_window_and_tracks_its_bounds(panel, monkeypatch):
    dashboard, session, overlay = panel
    target = Mock(title="Test Game")
    target.bounds.return_value = (10, 20, 200, 300)
    target.is_foreground.return_value = True
    monkeypatch.setattr(dashboard, "_create_window_live_session", Mock(return_value=session))
    dashboard.live_capture_picker.cget("command")(dashboard.live_capture_picker.cget("values")[1])
    dashboard._select_live_window(target)

    dashboard.btn_scan.invoke()

    dashboard._create_window_live_session.assert_called_once_with(target)
    overlay.prepare.assert_called_once_with((10, 20, 200, 300), exclude_from_capture=False)
    session.start.assert_called_once_with((10, 20, 200, 300))
    assert dashboard.btn_pick_window.cget("state") == "disabled"
    assert dashboard.btn_area.cget("state") == "normal"

    target.bounds.return_value = (-100, 20, 300, 400)
    dashboard._poll_live()
    session.start.assert_called_with((-100, 20, 300, 400))
    overlay.prepare.assert_called_with((-100, 20, 300, 400), exclude_from_capture=False)

    target.is_foreground.return_value = False
    dashboard._poll_live()
    overlay.set_visible.assert_called_with(False)
    target.bounds.return_value = None
    dashboard._poll_live()
    session.stop.assert_called_once()

    target.bounds.return_value = (-100, 20, 300, 400)
    target.is_foreground.return_value = True
    dashboard._poll_live()
    assert session.start.call_count == 3
    overlay.set_visible.assert_called_with(True)

    dashboard.btn_scan.invoke()
    dashboard._set_live_capture_mode("desktop")
    dashboard.btn_scan.invoke()
    session.start.assert_called_with(dashboard.fullscreen_region())
    overlay.prepare.assert_called_with(dashboard.fullscreen_region())


def test_fullscreen_reset_restarts_the_active_crop(panel):
    dashboard, session, overlay = panel
    crop = (105, 210, 325, 240)
    dashboard._set_area(crop)
    dashboard.btn_scan.invoke()
    session.start.assert_called_once_with(crop)

    dashboard.btn_fullscreen.invoke()

    assert session.start.call_args.args == (dashboard.fullscreen_region(),)
    assert overlay.prepare.call_args.args == (dashboard.fullscreen_region(),)
    assert dashboard.auto_switch.get() == 1


def test_recording_can_select_or_clear_a_region_inside_the_window(panel, monkeypatch):
    dashboard, session, _ = panel
    dashboard._set_live_capture_mode("window")
    target = Mock(title="Crop Game")
    target.bounds.return_value = (-300, 100, 200, 100)
    target.is_foreground.return_value = True
    dashboard._select_live_window(target)
    picker = Mock()
    monkeypatch.setattr("frontend.ui.RegionPicker", picker)
    monkeypatch.setattr(dashboard, "_create_window_live_session", Mock(return_value=session))

    dashboard.btn_scan.invoke()
    dashboard.btn_area.invoke()

    assert picker.call_args.kwargs["region"] == target.bounds.return_value
    picker.call_args.args[1]((-250, 125, 100, 50))
    assert dashboard._live_window_crop.box(400, 200) == (100, 50, 300, 150)
    assert dashboard.auto_switch.get() == 1
    assert dashboard.btn_fullscreen.cget("state") == "normal"

    dashboard.btn_fullscreen.invoke()
    assert dashboard._live_window_crop is None
    assert dashboard.auto_switch.get() == 1


@pytest.mark.parametrize("focus_allowed", [True, False])
def test_recording_shows_cropped_subtitles_after_restarting_from_dashboard(panel, monkeypatch, focus_allowed):
    dashboard, session, overlay = panel
    dashboard._set_live_capture_mode("window")
    region = (10, 20, 200, 300)
    target = Mock(title="Test Game")
    target.bounds.return_value = region
    target.is_foreground.return_value = True
    if focus_allowed:
        target.activate.side_effect = lambda: setattr(target.is_foreground, "return_value", True)
    else:
        target.activate.side_effect = OSError("Windows blocked foreground activation")
    monkeypatch.setattr(dashboard, "_create_window_live_session", Mock(return_value=session))
    dashboard._select_live_window(target)
    picker = Mock()
    monkeypatch.setattr("frontend.ui.RegionPicker", picker)
    dashboard.btn_area.invoke()
    picker.call_args.args[1]((30, 40, 100, 80))

    dashboard.btn_scan.invoke()
    dashboard.btn_scan.invoke()
    target.is_foreground.return_value = False  # Dashboard is foreground for the next Start click.
    dashboard.btn_scan.invoke()
    lines = (SimpleNamespace(source="source", translation="bản dịch"),)
    session.poll.return_value = SimpleNamespace(region=region, lines=lines, error=None)
    dashboard._poll_live()

    assert dashboard._live_window_crop.box(*region[2:]) == (20, 20, 120, 100)
    overlay.render.assert_called_once_with(region, lines)
    dashboard._create_window_live_session.assert_called_once_with(target)
    session.close.assert_not_called()
    target.activate.assert_called_once()
    if not focus_allowed:
        assert dashboard._status[0] == "status_live_focus_window"
        overlay.set_visible.assert_called_with(False)
        target.is_foreground.return_value = True
        session.poll.return_value = None
        dashboard._poll_live()
    overlay.set_visible.assert_called_with(True)
    assert dashboard._status[0] == "status_live_running"


def test_stop_hides_live_and_ignores_late_update(panel):
    dashboard, session, overlay = panel
    dashboard.btn_scan.invoke()
    session.poll.return_value = SimpleNamespace(
        region=dashboard.fullscreen_region(), lines=(object(),), error=None
    )

    dashboard.btn_scan.invoke()
    dashboard._poll_live()

    session.stop.assert_called_once()
    overlay.hide.assert_called()
    overlay.render.assert_not_called()
    assert dashboard.auto_switch.get() == 0


def test_overlay_coordinates_are_crop_local_and_scaled_to_canvas():
    from frontend.live_overlay import canvas_box

    assert canvas_box((10, 30, 60, 110), (450, 300, 200, 300), (100, 150)) == (
        5, 15, 30, 55
    )


@pytest.mark.parametrize("layout_pending", [False, True])
def test_dense_subtitles_keep_opaque_text_above_translucent_backgrounds(panel, monkeypatch, layout_pending):
    import win32con
    import win32gui

    from frontend.live_overlay import LiveOverlay

    dashboard, _, _ = panel
    overlay = LiveOverlay(dashboard)
    region = (25, 60, 320, 150)
    lines = (
        SimpleNamespace(box=(10, 15, 170, 37), translation="Dòng dịch dài cần xuống hàng để đọc hết."),
        SimpleNamespace(box=(10, 36, 170, 58), translation="Dòng kế tiếp"),
    )
    try:
        overlay.prepare(region)
        if layout_pending:
            monkeypatch.setattr(overlay.canvas, "winfo_width", lambda: 1)
            monkeypatch.setattr(overlay.canvas, "winfo_height", lambda: 1)
        else:
            dashboard.update()
        overlay.render(region, lines)

        assert [overlay.canvas.type(item) for item in overlay.canvas.find_all()] == ["text", "text"]
        backgrounds = overlay.background_canvas.find_all()
        assert [overlay.background_canvas.type(item) for item in backgrounds] == ["rectangle", "rectangle"]
        _, first_top, _, first_bottom = overlay.background_canvas.coords(backgrounds[0])
        _, second_top, _, second_bottom = overlay.background_canvas.coords(backgrounds[1])
        assert first_top > 0 and second_top > first_top
        assert first_top < second_bottom and second_top < first_bottom
        assert 0 < overlay.background_window.attributes("-alpha") < 1
        assert overlay.window.attributes("-alpha") == 1

        text_hwnd = win32gui.GetAncestor(overlay.window.winfo_id(), win32con.GA_ROOT)
        background_hwnd = win32gui.GetAncestor(overlay.background_window.winfo_id(), win32con.GA_ROOT)
        assert win32gui.GetWindowRect(text_hwnd) == win32gui.GetWindowRect(background_hwnd)
        assert win32gui.GetWindow(background_hwnd, win32con.GW_HWNDPREV) == text_hwnd

        overlay.hide()
        assert overlay.window.state() == overlay.background_window.state() == "withdrawn"
        assert not overlay.canvas.find_all() and not overlay.background_canvas.find_all()
    finally:
        overlay.close()


def test_live_does_not_start_if_visible_controls_cannot_be_excluded(panel, monkeypatch):
    dashboard, session, overlay = panel
    dashboard.update()

    def cannot_exclude(window, enabled=True):
        if enabled:
            raise OSError("capture exclusion unavailable")

    monkeypatch.setattr("frontend.ui.set_capture_exclusion", cannot_exclude)
    dashboard.trigger_scan()

    session.start.assert_not_called()
    overlay.hide.assert_called()
    assert dashboard.auto_switch.get() == 0
    assert "capture exclusion unavailable" in dashboard.status_label.cget("text")
