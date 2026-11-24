from types import SimpleNamespace
from unittest.mock import Mock
from time import monotonic, sleep

import keyboard
import pytest

from frontend.ui import ControlPanel


@pytest.fixture(scope="module")
def gui():
    # Keep one Tcl/Tk root per test process for these dashboard cases.
    with pytest.MonkeyPatch.context() as patch:
        from core.ocr_settings import OCRSettings
        from frontend import ui

        def webview_unavailable(*_args):
            raise OSError("WebView2 unavailable")

        patch.setattr(ui, "WebSettingsWindow", webview_unavailable)
        patch.setattr(ui, "available_model_settings", lambda _root: OCRSettings())
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


def test_live_capture_fps_uses_the_environment_setting(monkeypatch):
    from core import live_translation

    create_session = Mock()
    monkeypatch.setattr(live_translation, "LiveTranslationSession", create_session)
    monkeypatch.setenv("LIVE_CAPTURE_FPS", "60")
    dashboard = SimpleNamespace(_ocr_settings=Mock(), _live_session_crops=None, _session_crops=lambda: ())

    ControlPanel._create_live_session(dashboard)

    assert create_session.call_args.kwargs["capture_fps"] == 60


def test_dashboard_uses_live_trans_name_and_transparent_assets(panel):
    dashboard, _, _ = panel
    assert dashboard.title() == "Live Trans"
    assert dashboard.logo_label.cget("text") == "Live Trans"
    for widget in (dashboard.logo_label, dashboard.btn_pick_window, dashboard.btn_settings):
        im = widget.cget("image").cget("light_image")
        assert im.mode == "RGBA"
        assert im.getchannel("A").getextrema() == (0, 255)


def test_token_card_shows_latest_qwen_usage_to_the_right_of_auto_translate(panel):
    from core.live_translation import LiveUpdate
    from core.ocr_settings import OCRSettings

    dashboard, session, overlay = panel
    previous_settings = dashboard._ocr_settings
    try:
        dashboard._apply_ocr_settings(OCRSettings(flow="direct", qwen_context=3000))
        dashboard.update_idletasks()
        assert dashboard.token_card.master is dashboard.auto_card.master
        assert dashboard.token_card.winfo_x() >= dashboard.auto_card.winfo_x() + dashboard.auto_card.winfo_width()
        assert dashboard.token_percent.cget("text") == "0%"
        dashboard._live_session = session
        dashboard._live_overlay = overlay
        dashboard._live_region = region = (0, 0, 100, 100)
        for count, text, color in [(2255, "75%", "#a6e3a1"), (2400, "80%", "#f38ba8")]:
            session.poll.return_value = LiveUpdate(region, (), token_count=count)
            dashboard._poll_live()
            assert dashboard.token_percent.cget("text") == text
            assert dashboard.token_bar.cget("progress_color") == color
            assert dashboard.token_bar.get() == pytest.approx(count / 3000)
        dashboard._show_token_tooltip()
        assert dashboard.token_tooltip.cget("text") == "2400/3000 token"
        assert dashboard.token_tooltip.winfo_manager() == "place"
        dashboard._hide_token_tooltip()
        assert not dashboard.token_tooltip.winfo_manager()
        dashboard._stop_live()
        assert dashboard.token_percent.cget("text") == "0%"
    finally:
        dashboard._apply_ocr_settings(previous_settings)


def test_dashboard_groups_region_and_context_actions_in_their_cards(panel):
    from core.ocr_settings import OCRSettings

    dashboard, _, _ = panel
    previous_settings = dashboard._ocr_settings
    try:
        dashboard._apply_ocr_settings(OCRSettings(flow="direct", history_turns=2))
        dashboard.update_idletasks()
        area_buttons = {
            button.cget("text"): button
            for button in dashboard.area_card.winfo_children()
            if button is not dashboard.area_info
        }
        preview = next(
            (area_buttons[label] for label in ("Xem vùng", "View regions") if label in area_buttons),
            None,
        )
        assert preview is not None
        assert preview.winfo_y() == dashboard.btn_fullscreen.winfo_y()
        assert preview.winfo_x() + preview.winfo_width() <= dashboard.btn_fullscreen.winfo_x()
        assert preview.cget("state") == "disabled"
        dashboard._set_area((10, 20, 100, 80))
        assert preview.cget("state") == "normal"

        context = next(
            child for child in dashboard.token_card.winfo_children()
            if isinstance(child, type(dashboard.btn_fullscreen))
        )
        assert context.winfo_ismapped()
        dashboard._apply_ocr_settings(OCRSettings(flow="direct", history_turns=0))
        dashboard.update_idletasks()
        assert not context.winfo_ismapped()
        dashboard._apply_ocr_settings(OCRSettings())
        dashboard.update_idletasks()
        assert preview.winfo_ismapped()
        assert not context.winfo_ismapped()
    finally:
        dashboard._apply_ocr_settings(previous_settings)


def test_token_card_only_uses_dashboard_space_with_qwen_presets(panel):
    from core.ocr_settings import PRESETS

    dashboard, _, _ = panel
    previous_settings = dashboard._ocr_settings
    try:
        for qwen_preset, pivot_preset in [("High", "Flash"), ("Ultra", "Medium")]:
            dashboard._apply_ocr_settings(PRESETS[qwen_preset])
            dashboard.update_idletasks()
            assert dashboard.token_card.winfo_manager() == "grid"
            qwen_auto_width = dashboard.auto_card.winfo_width()
            dashboard._show_token_tooltip()

            dashboard._apply_ocr_settings(PRESETS[pivot_preset])
            dashboard.update_idletasks()
            assert not dashboard.token_card.winfo_manager()
            assert not dashboard.token_tooltip.winfo_manager()
            assert dashboard.auto_card.winfo_width() > qwen_auto_width
    finally:
        dashboard._apply_ocr_settings(previous_settings)


@pytest.mark.parametrize("ready", [True, False])
def test_settings_ultra_saves_only_when_required_items_are_ready(panel, tmp_path, monkeypatch, ready):
    from core.ocr_settings import OCRSettings, PRESETS
    from frontend import ocr_settings_window, ui

    dashboard, session, _ = panel
    item = SimpleNamespace(label="Qwen GGUF", is_ready=lambda: ready)
    inventory = Mock(side_effect=lambda _root, settings: [item] if settings.flow == "direct" else [])
    monkeypatch.setattr(ocr_settings_window, "model_items", inventory)
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())

    dashboard.trigger_scan()
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    dialog._choose_preset(3)
    assert dialog.preset_name.cget("text") == "Ultra"
    assert dialog.det_menu.get() == dialog.rec_menu.get() == "PP-OCRv6 Medium"
    dialog.save_button.invoke()

    if not ready:
        assert "Qwen GGUF" in dialog.status.cget("text")
        assert "Thiết lập" in dialog.status.cget("text")
        assert dialog.winfo_exists()
        assert OCRSettings.load(tmp_path) == dashboard._ocr_settings == OCRSettings()
        session.close.assert_not_called()
        dialog.destroy()
        return

    deadline = monotonic() + 3
    while dialog.winfo_exists() and monotonic() < deadline:
        dashboard.update()
        sleep(0.01)
    assert not dialog.winfo_exists()
    inventory.assert_called_with(tmp_path, PRESETS["Ultra"])
    assert OCRSettings.load(tmp_path) == dashboard._ocr_settings == PRESETS["Ultra"]
    session.close.assert_called_once()
    assert dashboard._live_session is None


def test_native_translation_color_updates_overlay_and_history(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui
    from frontend.live_overlay import LiveOverlay

    dashboard, session, _ = panel
    saved_settings = OCRSettings()
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", saved_settings)
    pick_color = Mock(return_value=((255, 0, 255), "#ff00ff"))
    monkeypatch.setattr(ocr_settings_window, "askcolor", pick_color)
    overlay = LiveOverlay(dashboard)
    monkeypatch.setattr(dashboard, "_live_overlay", overlay)
    dashboard._live_session = session
    overlay.region = region = (0, 0, 320, 150)
    lines = (SimpleNamespace(box=(10, 15, 170, 37), translation="Bản dịch"),)
    overlay.render(region, lines)
    item = overlay.canvas.find_all()[0]
    dashboard.add_history_log("Source", "Bản dịch đã có")

    def assert_history_color(color):
        history = dashboard.history_box
        assert history.tag_cget("translation", "foreground") == color
        assert history.get(*history.tag_ranges("translation")[-2:]) == "Bản dịch đã có"

    try:
        dashboard.btn_settings.invoke()
        dialog = dashboard._settings_window
        dialog.translation_color_button.invoke()
        pick_color.assert_called_once()
        dialog.save_button.invoke()
        assert OCRSettings.load(tmp_path).translation_color == "#ff00ff"
        assert overlay.canvas.itemcget(item, "fill") == "#ff00ff"
        assert str(overlay.window.attributes("-transparentcolor")).lower() == "#00ff00"
        assert_history_color("#ff00ff")
        session.close.assert_not_called()

        dashboard.btn_settings.invoke()
        dialog = dashboard._settings_window
        dialog.translation_color_reset.invoke()
        dialog.save_button.invoke()
        assert OCRSettings.load(tmp_path).translation_color is None
        assert overlay.canvas.itemcget(item, "fill") == "#f8fafc"
        assert_history_color("")
    finally:
        if dashboard._settings_window is not None and dashboard._settings_window.winfo_exists():
            dashboard._settings_window.destroy()
        overlay.close()


def test_native_settings_accept_typed_history_turns(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings(flow="direct", history_turns=3))
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    try:
        assert dialog.history_entry.get() == "3"
        for invalid in ("", "51", "-1", "2.5"):
            dialog.history_entry.delete(0, "end")
            dialog.history_entry.insert(0, invalid)
            dialog.save_button.invoke()
            assert dialog.winfo_exists()
            assert "0" in dialog.status.cget("text") and "50" in dialog.status.cget("text")
        dialog.history_entry.delete(0, "end")
        dialog.history_entry.insert(0, "50")
        dialog.save_button.invoke()
        assert not dialog.winfo_exists()
        assert OCRSettings.load(tmp_path).history_turns == dashboard._ocr_settings.history_turns == 50
    finally:
        if dialog.winfo_exists():
            dialog.destroy()


def test_native_advanced_qwen_backend_survives_preset_switches(panel, tmp_path, monkeypatch):
    from core.live_models import LiveModelError
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    applied = Mock()
    probe = Mock(side_effect=LiveModelError("CUDA 13 libraries are missing."))
    monkeypatch.setattr(ocr_settings_window, "check_qwen_cuda_worker", probe)
    dialog = ocr_settings_window.OCRSettingsWindow(dashboard, tmp_path, OCRSettings(), "VI", applied)
    try:
        dialog._toggle_advanced()
        assert not dialog.context_frame.winfo_manager()

        dialog._choose_preset(3)
        assert dialog.context_frame.winfo_manager() == "pack"
        assert dialog.backend_menu.get() == "llama.cpp Vulkan (mặc định)"
        dialog.backend_menu.set("llama-cpp-python CUDA (bổ sung)")

        dialog._choose_preset(1)
        assert not dialog.context_frame.winfo_manager()
        dialog._choose_preset(2)
        assert dialog.backend_menu.get() == "llama-cpp-python CUDA (bổ sung)"
        dialog.save_button.invoke()
        assert dialog.winfo_exists()
        assert "CUDA 13 libraries" in dialog.status.cget("text")
        assert OCRSettings.load(tmp_path).qwen_backend == "vulkan"
        applied.assert_not_called()

        probe.side_effect = None
        dialog.save_button.invoke()
        saved = OCRSettings.load(tmp_path)
        assert saved.preset == "High"
        assert saved.qwen_backend == "cuda-python"
        applied.assert_called_once_with(saved)
    finally:
        if dialog.winfo_exists():
            dialog.destroy()


def test_settings_interface_defaults_to_native_and_can_switch_to_web(panel, tmp_path, monkeypatch):
    from frontend import ocr_settings_window, ui
    from core.ocr_settings import OCRSettings

    dashboard, _, _ = panel
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    stop_live = Mock()
    monkeypatch.setattr(dashboard, "_stop_live", stop_live)
    host = Mock()
    monkeypatch.setattr(ui, "WebSettingsWindow", host)
    dashboard.btn_settings.invoke()
    native = dashboard._settings_window
    host.assert_not_called()
    assert native.interface_menu.get() == "Tối giản"
    native.interface_menu.set("Nâng cao")
    native.save_button.invoke()
    assert OCRSettings.load(tmp_path).settings_interface == "advanced"
    stop_live.assert_not_called()
    dashboard.btn_settings.invoke()
    host.assert_called_once()
    host.call_args.args[3]()


def test_dashboard_uses_one_web_settings_window_and_falls_back_on_startup_failure(panel, tmp_path, monkeypatch):
    from frontend import ui
    from frontend.ocr_settings_window import OCRSettingsWindow
    from core.ocr_settings import OCRSettings

    dashboard, _, _ = panel
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings(settings_interface="advanced"))
    window = Mock()
    constructor = Mock(return_value=window)
    monkeypatch.setattr(ui, "WebSettingsWindow", constructor)
    dashboard.btn_settings.invoke()
    dashboard.btn_settings.invoke()
    constructor.assert_called_once()
    window.focus_set.assert_called_once()

    _, request, fallback, closed = constructor.call_args.args
    reply = Mock()
    request("bootstrap", None, reply)
    assert len(reply.call_args.args[0]["presets"]) == 6
    fallback()
    closed()
    assert isinstance(dashboard._settings_window, OCRSettingsWindow)
    assert not dashboard.attributes("-disabled")
    dashboard._settings_window.destroy()


def test_web_rules_quick_close_replies_and_cancels_window_timers(panel, tmp_path):
    from frontend.web_settings import WebSettingsSession

    dashboard, _, _ = panel
    session = WebSettingsSession(dashboard, tmp_path, dashboard._ocr_settings, "VI", Mock())
    reply = Mock()
    session.request("manage_rules", None, reply)
    editor = session._rules_window
    timers = {
        job for job in dashboard.tk.call("after", "info")
        if dashboard.tk.call("after", "info", job)[0] in editor._tclCommands
    }
    editor.destroy()
    reply.assert_called_once()
    assert session._rules_window is None
    assert dashboard.grab_current() is None
    assert not timers.intersection(dashboard.tk.call("after", "info"))


@pytest.mark.parametrize("preset", [None, "Ultra", "Extreme", "Ultimate"])
def test_web_rule_editor_uses_the_draft_preset_colors(panel, tmp_path, preset):
    from frontend.ocr_settings_window import _PRESET_APPEARANCES
    from frontend.web_settings import WebSettingsSession

    dashboard, _, _ = panel
    session = WebSettingsSession(dashboard, tmp_path, dashboard._ocr_settings, "VI", Mock())
    session.request("manage_rules", {"theme_preset": preset}, Mock())
    try:
        editor = session._rules_window
        if preset:
            theme = _PRESET_APPEARANCES[preset]
            assert editor.heading.cget("text_color") == theme.ink
            assert editor.slot_menu.cget("fg_color") == theme.control
            assert editor.name_entry.cget("border_color") == theme.border
            assert editor.save_button.cget("fg_color") == theme.control
        else:
            assert editor.save_button.cget("fg_color") == editor.slot_menu.cget("fg_color")
    finally:
        session.close()


def test_dashboard_stays_blocked_after_closing_web_rules(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings(settings_interface="advanced"))
    mock_window = Mock()
    host = Mock(return_value=mock_window)
    monkeypatch.setattr(ui, "WebSettingsWindow", host)
    start_live = Mock()
    monkeypatch.setattr(dashboard, "_start_live", start_live)
    dashboard.btn_settings.invoke()
    _, request, _, closed = host.call_args.args
    try:
        reply = Mock()
        request("manage_rules", None, reply)
        rule_editor = dashboard.grab_current()
        rule_editor.tk.call(rule_editor.protocol("WM_DELETE_WINDOW"))
        reply.assert_called_once()
        assert dashboard._settings_window is mock_window
        assert dashboard.attributes("-disabled")
        dashboard.trigger_scan()
        start_live.assert_not_called()
    finally:
        closed()
    assert not dashboard.attributes("-disabled")
    dashboard.trigger_scan()
    start_live.assert_called_once()


def test_advanced_direct_flow_retains_context_and_marks_a_custom_preset(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    dialog.advanced_button.invoke()
    dialog._choose_preset(3)
    dialog.det_menu.set("PP-OCRv6 Small")
    dialog._update_selection()
    assert dialog.preset_name.cget("text") == "Tùy chỉnh"
    assert dialog.context_entry.winfo_manager()
    dialog.context_text.set("25001")
    dialog.save_button.invoke()
    assert dialog.winfo_exists()
    assert OCRSettings.load(tmp_path).flow == "pivot"
    dialog.context_text.set("5200")
    dialog.save_button.invoke()
    assert not dialog.winfo_exists()
    assert OCRSettings.load(tmp_path) == OCRSettings(rec="medium", flow="direct", qwen_context=5200)


def test_preset_slider_has_six_stops_and_thumb_reaches_both_ends(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    try:
        slider = dialog.slider
        canvas = slider.canvas
        deadline = monotonic() + 2
        while not canvas.winfo_ismapped() and monotonic() < deadline:
            dashboard.update()
            sleep(0.01)
        assert canvas.winfo_ismapped()
        assert [button.cget("text") for button in dialog.preset_buttons] == [
            "Flash", "Medium", "High", "Ultra", "Extreme", "Ultimate",
        ]
        assert len(canvas.find_withtag("preset-marker")) == 6

        def track_bounds():
            track_x = [
                x for item in canvas.find_withtag("preset-track") for x in canvas.coords(item)[::2]
            ]
            return min(track_x), max(track_x)

        def thumb_bounds():
            left, _, right, _ = canvas.coords("preset-thumb")
            return left, right

        left, right = track_bounds()
        assert thumb_bounds()[0] == pytest.approx(left, abs=1)

        y = canvas.winfo_height() // 2
        dialog.interface_menu.set("Nâng cao")
        dialog.animation_switch.toggle()
        _, _, start, travel = slider._geometry()

        def drag_to(position):
            canvas.event_generate("<B1-Motion>", x=round(start + travel * position / 5), y=y)

        canvas.event_generate("<Button-1>", x=int(left), y=y)
        drag_to(2.6)
        assert slider.get() == 2
        canvas.event_generate("<ButtonRelease-1>", x=round(start + travel * 2.6 / 5), y=y)
        assert slider.get() == 2
        canvas.event_generate("<Button-1>", x=round(start + travel * 2 / 5), y=y)
        drag_to(2.8)
        assert slider.get() == 3
        drag_to(3.6)
        assert slider.get() == 3
        drag_to(3.8)
        assert slider.get() == 4
        drag_to(4.8)
        assert slider.get() == 4
        canvas.event_generate("<ButtonRelease-1>", x=round(start + travel * 4.8 / 5), y=y)

        canvas.event_generate("<Button-1>", x=int(left), y=y)
        canvas.event_generate("<B1-Motion>", x=int(right), y=y)
        canvas.event_generate("<ButtonRelease-1>", x=int(right), y=y)
        assert slider.get() == 5
        assert dialog.preset_name.cget("text") == "Ultimate"
        assert "Qwen3.8-27B" in dialog.summary.cget("text")
        assert "Nhật → Việt · Anh → Việt" in dialog.summary.cget("text")

        deadline = monotonic() + 1
        while thumb_bounds()[1] < right - 1 and monotonic() < deadline:
            dashboard.update()
            sleep(0.01)
        assert thumb_bounds()[1] == pytest.approx(right, abs=1)

        dialog.preset_buttons[4].invoke()
        assert slider.get() == 4
        assert dialog.preset_name.cget("text") == "Extreme"
    finally:
        dialog.destroy()


def test_settings_tier_text_colors_follow_preset_and_reset(panel, tmp_path, monkeypatch):
    from colorsys import rgb_to_hsv
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    try:
        def text_hsv(widget):
            color = widget.cget("text_color")
            if isinstance(color, (tuple, list)):
                color = color[1]  # Settings uses dark appearance.
            rgb = tuple(component / 65535 for component in widget.winfo_rgb(color))
            return rgb_to_hsv(*rgb)

        def colors():
            labels = (dialog.title_label, dialog.preset_name, dialog.summary,
                      *dialog._field_labels.values(), dialog.context_label, dialog.active_rule_label)
            return tuple(text_hsv(label) for label in labels)

        dialog.interface_menu.set("Nâng cao")
        dialog.animation_switch.toggle()
        neutral = None
        for index, family in [(0, None), (1, None), (2, None), (3, (0.70, 0.88)),
                              (4, (0.24, 0.48)), (5, (0.04, 0.16)), (0, None)]:
            dialog.preset_buttons[index].invoke()
            assert dialog.preset_name.winfo_manager()
            current = colors()
            assert all(value >= 0.55 for _, _, value in current)
            if family is None:
                if neutral is None:
                    neutral = current
                assert current == neutral
            else:
                assert all(family[0] <= hue <= family[1] and saturation >= 0.1
                           for hue, saturation, _ in current)

        dialog.preset_buttons[3].invoke()
        ultra = colors()
        dialog.det_menu.set("PP-OCRv6 Small")
        dialog._update_selection()
        assert dialog.preset_name.cget("text") == "Tùy chỉnh"
        assert colors() == ultra
        dialog.flow_menu.set("Nhật → Anh → Việt")
        dialog._update_selection()
        assert colors() == neutral
    finally:
        dialog.destroy()


def test_settings_cancels_tier_animation_when_closed(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    try:
        dialog.preset_buttons[3].invoke()
        dialog.interface_menu.set("Nâng cao")
        dialog.animation_switch.toggle()
        deadline = monotonic() + 2
        while not dialog._appearance_job and monotonic() < deadline:
            dashboard.update()
            sleep(0.01)
        job = dialog._appearance_job
        assert job
        assert job in dashboard.tk.call("after", "info")
        dialog.destroy()
        assert job not in dashboard.tk.call("after", "info")
    finally:
        if dialog.winfo_exists():
            dialog.destroy()


@pytest.mark.parametrize("index", [0, 1])
def test_flash_and_medium_lock_pivot_while_allowing_advanced_customization(panel, tmp_path, monkeypatch, index):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    dialog._choose_preset(index)
    assert dialog.flow_menu.cget("state") == "disabled"
    dialog.det_menu.set("PP-OCRv6 Medium" if index == 0 else "PP-OCRv6 Small")
    dialog.ja_menu.set("NLLB-200 Distilled 600M" if index == 0 else "Mitsua ElanMT")
    dialog._update_selection()
    assert dialog.flow_menu.cget("state") == "disabled"
    assert dialog._selection().flow == "pivot"
    dialog.save_button.invoke()
    saved = OCRSettings.load(tmp_path)
    assert saved.flow == "pivot"
    assert saved.det == ("medium" if index == 0 else "small")
    assert saved.ja_model == ("nllb" if index == 0 else "mitsua")


@pytest.mark.parametrize("index, model, label", [
    (2, "2b", "Qwen3.5-2B Q4_K_M"),
    (3, "4b", "Qwen3.5-4B Q4_K_M"),
    (4, "9b", "Qwen3.5-9B Q4_K_M"),
    (5, "27b", "Qwen3.8-27B Q4_K_M"),
])
def test_qwen_preset_preserves_variant_through_advanced_and_rule_changes(panel, tmp_path, monkeypatch, index, model, label):
    from core.ocr_settings import OCRSettings
    from core.qwen_rules import QwenRule
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    dialog._choose_preset(index)
    assert label in dialog.summary.cget("text")
    assert label in dialog.context_label.cget("text")
    assert dialog.flow_menu.cget("state") == "normal"
    dialog.flow_menu.set("Nhật → Anh → Việt")
    dialog._update_selection()
    dialog.flow_menu.set("Nhật → Việt · Anh → Việt")
    dialog.det_menu.set("PP-OCRv6 Small")
    dialog._select_rule(QwenRule("a" * 32, "Rule", "Text"))
    dialog.context_text.set("5200")
    dialog._update_selection()
    dialog.save_button.invoke()
    saved = OCRSettings.load(tmp_path)
    assert saved.qwen_model == model
    assert saved.qwen_context == 5200
    assert saved.qwen_rule == QwenRule("a" * 32, "Rule", "Text")


def test_matching_medium_by_advanced_edits_locks_the_flow(panel, tmp_path, monkeypatch):
    from core.ocr_settings import PRESETS
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", PRESETS["High"])
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    try:
        dialog.flow_menu.set("Nhật → Anh → Việt")
        dialog.rec_menu.set("PP-OCRv6 Small")
        dialog.ja_menu.set("NLLB-200 Distilled 600M")
        dialog._update_selection()
        assert dialog.preset_name.cget("text") == "Medium"
        assert dialog.flow_menu.cget("state") == "disabled"
    finally:
        dialog.destroy()


def test_high_missing_model_blocks_save_with_warning(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from frontend import ocr_settings_window, ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings())
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda _root, settings: [
        SimpleNamespace(label="Qwen 2B", is_ready=lambda: False)
    ] if settings.qwen_model == "2b" and settings.flow == "direct" else [])
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    dialog._choose_preset(2)
    dialog.save_button.invoke()
    assert dialog.winfo_exists()
    assert "Qwen 2B" in dialog.status.cget("text")
    assert OCRSettings.load(tmp_path) == OCRSettings()
    dialog.destroy()


def test_sidebar_setup_offers_missing_nllb_after_medium_save_is_blocked(panel, tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk

    from core import setup_models
    from core.ocr_settings import PRESETS
    from frontend import ui

    dashboard, _, _ = panel
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", PRESETS["Flash"])
    monkeypatch.setattr(setup_models, "_ready_path", lambda root, env, model: (
        None if model.key == "translate-ja-en-nllb" else tmp_path
    ))

    dashboard.btn_settings.invoke()
    settings = dashboard._settings_window
    try:
        settings._choose_preset(1)
        settings.save_button.invoke()
        assert "NLLB" in settings.status.cget("text")
        assert dashboard._ocr_settings == PRESETS["Flash"]
    finally:
        settings.destroy()

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    deadline = monotonic() + 3
    offered = False

    def check_setup():
        nonlocal offered, timer
        window = dashboard.grab_current()
        if isinstance(window, tk.Toplevel):
            controls = list(descendants(window))
            optional_group = any(isinstance(widget, ttk.Label) and widget.winfo_ismapped()
                                 and widget.cget("text") == "Mô hình bổ sung" for widget in controls)
            addon = any(isinstance(widget, ttk.Checkbutton) and widget.winfo_ismapped()
                        and widget.instate(["!disabled"]) and "NLLB" in widget.cget("text")
                        for widget in controls)
            continue_button = next((widget for widget in controls if isinstance(widget, ttk.Button)
                                    and widget.winfo_ismapped() and widget.instate(["!disabled"])
                                    and widget.cget("text") == "Tiếp tục"), None)
            if optional_group and addon and continue_button is not None:
                offered = True
                continue_button.invoke()
                return
            if monotonic() >= deadline:
                window.tk.call(window.protocol("WM_DELETE_WINDOW"))
                return
        timer = dashboard.after(10, check_setup)

    timer = dashboard.after(10, check_setup)
    try:
        dashboard.btn_setup.invoke()
    finally:
        dashboard.after_cancel(timer)
    assert offered, "Setup closed without offering the missing NLLB model"


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


def test_translate_once_is_a_mode_and_scan_ignores_unrelated_updates(panel, monkeypatch):
    from core.live_translation import LiveUpdate, TranslatedLine
    from core.ocr_settings import OCRSettings

    dashboard, session, overlay = panel
    previous_settings = dashboard._ocr_settings
    monkeypatch.setattr("frontend.ui.model_items", lambda *_args: ())
    dashboard._apply_ocr_settings(OCRSettings(flow="direct", qwen_context=3000))
    session.close.reset_mock()
    session.request_once.side_effect = (11, 12)
    try:
        picker = dashboard.live_capture_picker
        desktop, _, once = picker.cget("values")
        picker.cget("command")(once)
        assert picker.get() == once
        assert dashboard._picker is None
        session.start.assert_not_called()
        session.request_once.assert_not_called()

        dashboard.btn_scan.invoke()
        region = dashboard.fullscreen_region()
        session.request_once.assert_called_once_with(region)
        assert dashboard.auto_switch.get() == 1
        assert dashboard._status[0] == "status_once_busy"
        assert dashboard.btn_scan.cget("state") == "disabled"
        dashboard.trigger_scan()
        session.request_once.assert_called_once()
        line = TranslatedLine((0, 0, 30, 10), "First", "VI First")

        for unrelated_id in (None, 10):
            session.poll.return_value = LiveUpdate(region, (line,), request_id=unrelated_id)
            dashboard._poll_live()
        overlay.render.assert_not_called()

        session.poll.return_value = LiveUpdate(region, (line,), token_count=2255, request_id=11)
        dashboard._poll_live()
        overlay.render.assert_called_once_with(region, (line,))
        assert dashboard.btn_scan.cget("state") == "normal"
        assert dashboard.auto_switch.get() == 0
        assert dashboard.token_percent.cget("text") == "75%"
        assert dashboard.token_tooltip.cget("text") == "2255/3000 token"
        assert "VI First" in dashboard.history_box.get("1.0", "end")

        dashboard.auto_switch.toggle()
        assert session.request_once.call_count == 2
        session.poll.return_value = LiveUpdate(region, (), request_id=12)
        dashboard._poll_live()
        overlay.render.assert_called_with(region, ())
        session.close.assert_not_called()
        assert dashboard._status[0] == "status_once_empty"
        assert dashboard.btn_scan.cget("state") == "normal"
        assert picker.get() == once
        picker.cget("command")(desktop)
        dashboard.btn_scan.invoke()
        session.start.assert_called_once_with(region)
    finally:
        session.request_once.side_effect = None
        dashboard._apply_ocr_settings(previous_settings)


def test_translate_once_window_picker_and_retained_overlay(panel, monkeypatch):
    from core.live_translation import LiveUpdate, TranslatedLine

    dashboard, session, overlay = panel
    target = Mock(title="Once Game", pid=123)
    target.bounds.return_value = region = (10, 20, 200, 300)
    target.is_foreground.return_value = True
    monkeypatch.setattr(dashboard, "_live_window", None)
    monkeypatch.setattr("frontend.ui.list_windows", lambda _pid: [target])
    monkeypatch.setattr(dashboard, "_create_window_live_session", Mock(return_value=session))
    dashboard._set_live_capture_mode("window")
    session.request_once.return_value = 42

    capture_picker = dashboard.live_capture_picker
    capture_picker.cget("command")(capture_picker.cget("values")[2])
    assert dashboard._picker is None
    dashboard.btn_scan.invoke()
    assert dashboard._status[0] == "status_pick_window"
    assert dashboard._picker is None
    dashboard.btn_pick_window.invoke()
    picker = dashboard._picker
    assert picker is not None
    session.request_once.assert_not_called()
    picker.tk.call(picker.protocol("WM_DELETE_WINDOW"))
    assert dashboard._picker is None
    dashboard.btn_pick_window.invoke()
    picker = dashboard._picker

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    next(widget for widget in descendants(picker) if isinstance(widget, type(dashboard.btn_scan))
         and widget.cget("text") == "Once Game  [123]").invoke()
    session.request_once.assert_not_called()
    dashboard.btn_scan.invoke()
    session.request_once.assert_called_once_with(region)
    line = TranslatedLine((0, 0, 20, 10), "Game text", "VI Game text")
    session.poll.return_value = LiveUpdate(region, (line,), request_id=42)
    dashboard._poll_live()
    overlay.render.assert_called_once_with(region, (line,))

    target.is_foreground.return_value = False
    target.bounds.return_value = (30, 40, 200, 300)
    dashboard._poll_live()
    overlay.move.assert_called_once_with((30, 40, 200, 300))
    overlay.set_visible.assert_called_with(False)
    target.is_foreground.return_value = True
    dashboard._poll_live()
    overlay.set_visible.assert_called_with(True)
    session.start.assert_not_called()
    session.request_once.assert_called_once()

    target.bounds.return_value = (30, 40, 300, 300)
    dashboard._poll_live()
    overlay.hide.assert_called()
    assert dashboard._once_region is None


def test_translate_once_waits_for_regions_and_drops_result_after_settings_change(panel, monkeypatch):
    from core.live_translation import LiveUpdate, TranslatedLine
    from core.ocr_settings import OCRSettings

    dashboard, session, overlay = panel
    previous = dashboard._ocr_settings
    picker = Mock()
    monkeypatch.setattr("frontend.ui.RegionPicker", picker)
    monkeypatch.setattr(dashboard, "_extra_crops", {"window": {}, "desktop": {}})
    session.request_once.return_value = 21
    try:
        dashboard._apply_ocr_settings(OCRSettings(crop_count=2))
        capture_picker = dashboard.live_capture_picker
        capture_picker.cget("command")(capture_picker.cget("values")[2])
        dashboard.btn_scan.invoke()
        assert dashboard._picker is None
        assert dashboard._status[0] == "status_live_error"
        dashboard.btn_area.invoke()
        picker.call_args.args[1](None)
        session.request_once.assert_not_called()
        dashboard.btn_area.invoke()
        picker.call_args.args[1]((10, 20, 100, 100))
        session.request_once.assert_not_called()
        picker.call_args.args[1]((120, 20, 100, 100))
        assert len(dashboard._crop_regions()) == 2
        session.request_once.assert_not_called()
        dashboard.btn_scan.invoke()
        region = dashboard.fullscreen_region()
        session.request_once.assert_called_once_with(region)
        assert len(dashboard._live_session_crops) == 2
        dashboard._apply_ocr_settings(OCRSettings())
        session.close.assert_called()
        line = TranslatedLine((0, 0, 25, 10), "Late text", "VI Late text")
        session.poll.return_value = LiveUpdate(region, (line,), request_id=21)
        dashboard._poll_live()
        overlay.render.assert_not_called()
        assert dashboard.btn_scan.cget("state") == "normal"
    finally:
        dashboard._apply_ocr_settings(previous)


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


def test_three_crops_are_selected_in_order_and_previewed_without_editing(panel, monkeypatch, tmp_path):
    from dataclasses import replace

    from core.qwen_rules import save_rule_slot

    dashboard, _, _ = panel
    monkeypatch.setattr(dashboard, "_ocr_settings", dashboard._ocr_settings)
    monkeypatch.setattr(dashboard, "_extra_crops", {"desktop": {}, "window": {}})
    monkeypatch.setattr(dashboard, "_desktop_area", None)
    monkeypatch.setattr("frontend.ui._ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "fullscreen_region", lambda: (0, 0, 1000, 800))
    rule = save_rule_slot(tmp_path, name="Character names", text="Vai trò: Tên người nói\nKeep official names.")
    dashboard._apply_ocr_settings(replace(
        dashboard._ocr_settings, crop_count=3, flow="direct",
        crop_rule_slots=(rule.id, "", ""),
    ))
    heading, note, *rows, _actions = dashboard.crop_panel.winfo_children()
    assert heading.cget("text") == "Quy tắc riêng từng vùng"
    assert "ưu tiên" in note.cget("text")
    for row in rows:
        assert len(row.winfo_children()) == 2
    rule_menu = rows[1].winfo_children()[1]
    assert rule_menu.get() == "Chỉ dùng quy tắc chung"
    picker = Mock()
    monkeypatch.setattr("frontend.ui.RegionPicker", picker)
    regions = ((10, 20, 140, 35), (30, 100, 300, 60), (40, 200, 320, 50))

    dashboard.start_area_selection()
    for index, region in enumerate(regions, 1):
        assert picker.call_args.args[2].startswith(f"Crop {index}")
        picker.call_args.args[1](region)
    crops = dashboard._session_crops()
    assert [crop.box(1000, 800) for crop, _, _ in crops] == [
        (x, y, x + width, y + height) for x, y, width, height in regions
    ]
    assert crops[0][1:] == ("speaker", "Keep official names.")
    assert crops[1][1:] == ("dialogue", "")
    dashboard.start_area_selection(preview=True)
    assert picker.call_args.kwargs["preview"] == tuple(
        (index, crop) for index, (crop, _, _) in enumerate(crops, 1)
    )
    picker.call_args.args[1](None)
    assert dashboard._session_crops() == crops


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

def test_qwen_rule_manager_saves_slots_and_applies_selected_snapshot(panel, tmp_path, monkeypatch):
    from core.ocr_settings import OCRSettings
    from core.qwen_rules import load_rule_slots
    from frontend import ocr_settings_window, ui

    dashboard, session, _ = panel
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_args: [])
    monkeypatch.setattr(ui, "_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_ocr_settings", OCRSettings(flow="direct"))
    dashboard.btn_settings.invoke()
    dialog = dashboard._settings_window
    dialog.advanced_button.invoke()
    dialog.manage_rules_button.invoke()
    manager = dialog._rules_window
    manager.name_text.set("My rule")
    manager.rule_text.insert("1.0", "Use concise wording")
    manager.save_button.invoke()
    assert load_rule_slots(tmp_path)[0].text == "Use concise wording"
    assert OCRSettings.load(tmp_path).qwen_rule is None
    manager.name_text.set("Renamed")
    manager.save_button.invoke()
    assert load_rule_slots(tmp_path)[0].name == "Renamed"
    assert "Renamed" in manager.heading.cget("text")
    assert "Đã lưu" in manager.status.cget("text")
    manager._select_slot(manager.no_rule_label)
    assert dialog._selection().qwen_rule is None
    manager._select_slot(manager.slot_labels[0])
    manager.destroy()
    assert dialog.grab_current() == dialog
    dialog.save_button.invoke()
    assert OCRSettings.load(tmp_path).qwen_rule == load_rule_slots(tmp_path)[0]
    assert dashboard._ocr_settings.qwen_rule == load_rule_slots(tmp_path)[0]



def test_window_move_repositions_subtitles_without_restarting(panel, monkeypatch):
    dashboard, session, overlay = panel
    dashboard._set_live_capture_mode("window")
    target = Mock(title="Test Game")
    target.bounds.return_value = (10, 20, 200, 300)
    target.is_foreground.return_value = True
    monkeypatch.setattr(dashboard, "_create_window_live_session", Mock(return_value=session))
    dashboard._select_live_window(target)
    dashboard._start_live()

    target.bounds.return_value = (-100, 20, 200, 300)
    dashboard._poll_live()

    session.start.assert_called_once_with((10, 20, 200, 300))
    session.move.assert_called_once_with((-100, 20, 200, 300))
    overlay.move.assert_called_once_with((-100, 20, 200, 300))


def test_restarting_with_an_edited_crop_rule_recreates_session(panel, tmp_path, monkeypatch):
    from dataclasses import replace

    from core.ocr_settings import PRESETS
    from core.qwen_rules import save_rule_slot

    dashboard, original_session, _ = panel
    monkeypatch.setattr("frontend.ui._ROOT", tmp_path)
    monkeypatch.setattr("frontend.ui.model_items", lambda *_: [])
    monkeypatch.setattr(dashboard, "_ocr_settings", dashboard._ocr_settings)
    slot = save_rule_slot(tmp_path, name="Crop rule", text="Role: Dialogue")
    dashboard._apply_ocr_settings(replace(PRESETS["High"], crop_rule_slots=(slot.id, "", "")))
    sessions = iter((original_session, Mock()))
    original_session.poll.return_value = None

    def create_session():
        session = next(sessions)
        session.poll.return_value = None
        session.role = dashboard._session_crops()[0][1]
        return session

    monkeypatch.setattr(dashboard, "_live_session_factory", Mock(side_effect=create_session))
    try:
        dashboard._start_live()
        assert dashboard._live_session.role == "dialogue"
        dashboard._stop_live()
        dashboard._start_live()
        assert dashboard._live_session is original_session
        dashboard._stop_live()

        save_rule_slot(tmp_path, name=slot.name, text="Role: Speaker", slot_id=slot.id)
        dashboard._start_live()

        assert dashboard._live_session.role == "speaker"
        original_session.close.assert_called_once()
    finally:
        dashboard._stop_live()
        dashboard._close_live_session()
