from types import SimpleNamespace
from unittest.mock import Mock

from core.ocr_settings import OCRSettings, PRESETS
from core.qwen_rules import QwenRule


def _draft(settings):
    return {key: getattr(settings, key) for key in
            ("det", "rec", "ja_model", "flow", "qwen_model", "qwen_context", "qwen_backend", "animation_enabled",
             "settings_interface", "crop_count", "history_turns", "translation_color", "post_translation_delay_ms")}


def _request(session, method, draft=None):
    reply = Mock()
    session.request(method, draft, reply)
    return reply.call_args.args[0]


def test_animation_defaults_off_for_existing_settings_and_persists(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    path = tmp_path / "models" / "live-ocr" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"flow": "direct", "qwen_model": "4b"}', encoding="utf-8")
    original = OCRSettings.load(tmp_path)
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    applied = Mock()
    session = WebSettingsSession(Mock(), tmp_path, original, "VI", applied)
    initial = _request(session, "bootstrap")["settings"]
    assert initial["animation_enabled"] is False
    assert initial["settings_interface"] == "minimal"
    draft = _draft(original) | {"animation_enabled": True, "settings_interface": "advanced"}
    assert _request(session, "validate", draft)["ok"]
    assert OCRSettings.load(tmp_path).animation_enabled is False
    assert _request(session, "save", draft)["ok"]
    saved = OCRSettings.load(tmp_path)
    assert saved.animation_enabled is True
    assert saved.settings_interface == "advanced"
    assert applied.call_count == 1
    reopened = WebSettingsSession(Mock(), tmp_path, saved, "VI", applied)
    assert _request(reopened, "bootstrap")["settings"]["animation_enabled"] is True
    assert _request(reopened, "save", draft | {"settings_interface": "minimal"})["ok"]
    saved = OCRSettings.load(tmp_path)
    assert saved.animation_enabled is True
    assert saved.settings_interface == "minimal"


def test_bootstrap_and_save_use_real_presets_and_rule_snapshot(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    rule = QwenRule("a" * 32, "A rule", "The saved rule text")
    original = OCRSettings(flow="direct", det="small", qwen_model="9b", qwen_rule=rule)
    applied = Mock()
    session = WebSettingsSession(Mock(), tmp_path, original, "EN", applied)

    boot = _request(session, "bootstrap")
    assert [preset["name"] for preset in boot["presets"]] == list(PRESETS)
    assert boot["settings"] == {**_draft(original), "rule_name": rule.name}
    assert boot["language"] == "EN"
    assert rule.text not in str(boot)

    draft = _draft(PRESETS["Ultimate"]) | {"qwen_context": "5200", "rule_name": "Ignored"}
    assert _request(session, "validate", draft) == {"ok": True, "message": "", "preset": "Ultimate"}
    assert OCRSettings.load(tmp_path) == OCRSettings()
    assert _request(session, "save", draft)["ok"]
    expected = OCRSettings(det="medium", rec="medium", flow="direct",
                           qwen_model="27b", qwen_context=5200, qwen_rule=rule)
    assert OCRSettings.load(tmp_path) == expected
    applied.assert_called_once_with(expected)


def test_missing_model_or_invalid_context_blocks_web_save(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    item = SimpleNamespace(label="Missing GGUF", is_ready=lambda: False)
    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [item])
    applied = Mock()
    session = WebSettingsSession(Mock(), tmp_path, OCRSettings(), "VI", applied)
    blocked = _request(session, "save", _draft(PRESETS["Ultra"]))
    assert not blocked["ok"] and "Missing GGUF" in blocked["message"]
    assert blocked["preset"] == "Ultra"
    invalid = _request(session, "save", _draft(PRESETS["High"]) | {"qwen_context": "NaN"})
    assert not invalid["ok"] and "context Qwen" in invalid["message"]
    assert OCRSettings.load(tmp_path) == OCRSettings()
    applied.assert_not_called()


def test_web_settings_keep_crop_choices_across_presets(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    original = OCRSettings(crop_count=2, history_turns=3,
                           crop_rule_slots=("a" * 32, "", ""))
    session = WebSettingsSession(Mock(), tmp_path, original, "EN", Mock())
    assert _request(session, "bootstrap")["settings"]["crop_count"] == 2
    assert _request(session, "bootstrap")["settings"]["history_turns"] == 3

    draft = _draft(PRESETS["Ultra"]) | {"crop_count": 3, "history_turns": 50}
    for valid in (0, 1, 2, 3, 20, 49, 50):
        assert _request(session, "validate", draft | {"history_turns": valid})["ok"]
    for invalid in (None, -1, 51, 2.5, True):
        result = _request(session, "save", draft | {"history_turns": invalid})
        assert not result["ok"]
        assert "0 to 50" in result["message"]
    assert _request(session, "save", draft)["ok"]
    saved = OCRSettings.load(tmp_path)
    assert (saved.crop_count, saved.history_turns) == (3, 50)
    assert saved.crop_rule_slots == original.crop_rule_slots


def test_web_translation_color_round_trips_without_a_pipeline_change(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    original = OCRSettings()
    inventory = Mock(return_value=[])
    monkeypatch.setattr(ocr_settings_window, "model_items", inventory)
    session = WebSettingsSession(Mock(), tmp_path, original, "EN", Mock())
    assert _request(session, "bootstrap")["settings"]["translation_color"] is None
    draft = _draft(original)
    assert not _request(session, "save", draft | {"translation_color": "blue"})["ok"]
    assert _request(session, "save", draft | {"translation_color": "#66ccff"})["ok"]
    saved = OCRSettings.load(tmp_path)
    assert saved.translation_color == "#66ccff"
    assert saved.same_pipeline_as(original)
    inventory.assert_not_called()
    assert _request(session, "save", draft | {"translation_color": None})["ok"]
    assert OCRSettings.load(tmp_path).translation_color is None


def test_web_selects_optional_cuda_without_changing_the_qwen_preset(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    monkeypatch.setattr(ocr_settings_window, "check_qwen_cuda_worker", Mock())
    applied = Mock()
    original = PRESETS["Ultra"]
    session = WebSettingsSession(Mock(), tmp_path, original, "VI", applied)
    boot = _request(session, "bootstrap")
    assert boot["settings"]["qwen_backend"] == "vulkan"
    assert boot["qwen_backends"]["cuda-python"] == "llama-cpp-python CUDA (bổ sung)"
    assert all("qwen_backend" not in preset for preset in boot["presets"])
    assert not _request(session, "validate", _draft(original) | {"qwen_backend": "unknown"})["ok"]
    assert _request(session, "save", _draft(original) | {"qwen_backend": "cuda-python"}) == {
        "ok": True, "message": "", "preset": "Ultra",
    }
    saved = OCRSettings.load(tmp_path)
    assert saved.qwen_backend == "cuda-python"
    applied.assert_called_once_with(saved)

    settings_without_backend = _draft(PRESETS["Flash"])
    settings_without_backend.pop("qwen_backend", None)
    assert _request(session, "save", settings_without_backend)["ok"]
    assert OCRSettings.load(tmp_path).qwen_backend == "cuda-python"



def test_incompatible_cuda_is_blocked_at_save_and_can_switch_to_vulkan(tmp_path, monkeypatch):
    import subprocess

    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    error = "llama-cpp-python CUDA needs an NVIDIA GPU and CUDA-enabled PyTorch."
    run = Mock(return_value=subprocess.CompletedProcess([], 1, stdout=error))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("MANGALENS_LIVE_PYTHON", "worker-python.exe")
    monkeypatch.setenv("LIVE_PYTHONPATH", "test-worker-models")

    original = PRESETS["High"]
    original.save(tmp_path)
    applied = Mock()
    session = WebSettingsSession(Mock(), tmp_path, original, "EN", applied)
    draft = _draft(original) | {"qwen_backend": "cuda-python"}
    assert _request(session, "validate", draft)["ok"]
    run.assert_not_called()
    result = _request(session, "save", draft)
    assert not result["ok"] and error in result["message"]
    assert OCRSettings.load(tmp_path).qwen_backend == "vulkan"
    applied.assert_not_called()
    assert run.call_args.args[0][0] == "worker-python.exe"
    assert run.call_args.kwargs["env"]["PYTHONPATH"].startswith("test-worker-models")

    assert _request(session, "save", _draft(original))["ok"]
    run.assert_called_once()


def test_post_translation_delay_round_trip_and_validation(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    monkeypatch.setattr(ocr_settings_window, "model_items", lambda *_: [])
    original = OCRSettings.load(tmp_path)
    assert original.post_translation_delay_ms == 0
    session = WebSettingsSession(Mock(), tmp_path, original, "EN", Mock())
    assert _request(session, "bootstrap")["settings"]["post_translation_delay_ms"] == 0
    draft = _draft(original)
    for invalid in (-1, 5001, 1.5, True, "", "1.5"):
        assert not _request(session, "save", draft | {"post_translation_delay_ms": invalid})["ok"]
    assert _request(session, "save", draft | {"post_translation_delay_ms": "5000"})["ok"]
    saved = OCRSettings.load(tmp_path)
    assert saved.post_translation_delay_ms == 5000
    assert saved.same_pipeline_as(original)
    assert saved.preset == original.preset


def test_invalid_saved_delay_defaults_without_discarding_other_settings(tmp_path):
    path = OCRSettings._path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"det": "medium", "post_translation_delay_ms": true}', encoding="utf-8")
    loaded = OCRSettings.load(tmp_path)
    assert loaded.det == "medium"
    assert loaded.post_translation_delay_ms == 0


def test_web_hot_settings_and_restart_context_save_without_rechecking_models(tmp_path, monkeypatch):
    from frontend import ocr_settings_window
    from frontend.web_settings import WebSettingsSession

    original = OCRSettings(flow="direct")
    inventory = Mock()
    monkeypatch.setattr(ocr_settings_window, "model_items", inventory)
    session = WebSettingsSession(Mock(), tmp_path, original, "EN", Mock())
    draft = _draft(original) | {"qwen_context": "8000", "history_turns": 4, "crop_count": 2}

    assert _request(session, "save", draft)["ok"]
    inventory.assert_not_called()
    saved = OCRSettings.load(tmp_path)
    assert (saved.qwen_context, saved.history_turns, saved.crop_count) == (8000, 4, 2)
