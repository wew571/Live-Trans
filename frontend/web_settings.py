"""Handle settings requests from the local WebView on the dashboard thread."""

from dataclasses import replace

from pywintypes import error as Win32Error

from core.ocr_settings import (
    DEFAULT_TRANSLATION_COLOR,
    FLOW_PIVOT,
    HISTORY_TURNS_MAX,
    PRESETS,
    QWEN_CONTEXT_MAX,
    QWEN_CONTEXT_MIN,
    QWEN_MODEL_LABELS,
    valid_history_turns,
)
from frontend.ocr_settings_window import _LABELS, _PRESET_APPEARANCES, _backend_options, selection_status
from frontend.qwen_rules_window import QwenRulesWindow
from frontend.windows_shell import place_window_beside, raise_settings_window


def _preset_fields(settings):
    return {
        "det": settings.det, "rec": settings.rec, "ja_model": settings.ja_model,
        "flow": settings.flow, "qwen_model": settings.qwen_model,
    }


class WebSettingsSession:
    def __init__(self, parent, root, settings, language, on_save):
        self._parent = parent
        self._root = root
        self._original = settings
        self._language = language
        self._texts = _LABELS[language]
        self._on_save = on_save
        self._selected_rule = settings.qwen_rule
        self._rules_window = None

    def request(self, method, draft, reply):
        if method == "bootstrap":
            reply(self._bootstrap())
        elif method == "manage_rules":
            self._manage_rules(draft, reply)
        elif method in {"validate", "save"}:
            reply(self._check(draft, save=(method == "save")))

    def close(self):
        if self._rules_window is not None:
            self._rules_window.destroy()

    def _rule_name(self):
        return self._selected_rule.name if self._selected_rule else None

    def _bootstrap(self):
        return {
            "language": self._language,
            "settings": {
                **_preset_fields(self._original),
                "qwen_context": self._original.qwen_context,
                "qwen_backend": self._original.qwen_backend,
                "animation_enabled": self._original.animation_enabled,
                "settings_interface": self._original.settings_interface,
                "crop_count": self._original.crop_count,
                "history_turns": self._original.history_turns,
                "translation_color": self._original.translation_color,
                "rule_name": self._rule_name(),
            },
            "presets": [{"name": name, **_preset_fields(preset)} for name, preset in PRESETS.items()],
            "qwen_models": QWEN_MODEL_LABELS,
            "qwen_backends": {backend: label for label, backend in _backend_options(self._texts).items()},
            "qwen_backend_label": self._texts["qwen_backend"],
            "qwen_backend_hint": self._texts["backend_hint"],
            "crop_labels": {
                "count": self._texts["crop_count"], "history": self._texts["history_turns"],
            },
            "history_turns_max": HISTORY_TURNS_MAX,
            "translation_color_labels": {
                "label": self._texts["translation_color"], "default": self._texts["default_color"],
            },
            "default_translation_color": DEFAULT_TRANSLATION_COLOR,
            "context_min": QWEN_CONTEXT_MIN,
            "context_max": QWEN_CONTEXT_MAX,
        }

    def _check(self, draft, *, save):
        try:
            flow = draft["flow"]
            history = draft.get("history_turns", self._original.history_turns)
            if not valid_history_turns(history):
                if flow != FLOW_PIVOT:
                    return {"ok": False, "message": self._texts["invalid_history"], "preset": None}
                history = self._original.history_turns
            raw_context = draft["qwen_context"]
            try:
                context = int(raw_context) if type(raw_context) in (int, str) else 0
            except ValueError:
                context = 0
            if not QWEN_CONTEXT_MIN <= context <= QWEN_CONTEXT_MAX:
                if flow != FLOW_PIVOT:
                    return {"ok": False, "message": self._texts["invalid_context"], "preset": None}
                context = self._original.qwen_context
            settings = replace(
                self._original, det=draft["det"], rec=draft["rec"], ja_model=draft["ja_model"],
                flow=flow, qwen_context=context, qwen_rule=self._selected_rule, qwen_model=draft["qwen_model"],
                qwen_backend=draft.get("qwen_backend", self._original.qwen_backend),
                animation_enabled=draft.get("animation_enabled", self._original.animation_enabled),
                settings_interface=draft.get("settings_interface", self._original.settings_interface),
                crop_count=draft.get("crop_count", self._original.crop_count),
                history_turns=history,
                translation_color=draft.get("translation_color", self._original.translation_color),
            )
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": self._texts["error"], "preset": None}

        message, _ = selection_status(self._root, settings, self._texts, original=self._original, saving=save)
        if not message and save and settings != self._original:
            try:
                settings.save(self._root)
            except OSError as error:
                message = f"{self._texts['error']}\n{error}"
            else:
                self._original = settings
                self._on_save(settings)
        return {"ok": not message, "message": message, "preset": settings.preset}

    def _manage_rules(self, draft, reply):
        if self._rules_window is not None:
            self._rules_window.focus_set()
            reply({"rule_name": self._rule_name()})
            return

        options = draft if isinstance(draft, dict) else {}
        settings_hwnd = options.get("settings_hwnd")
        preset = options.get("theme_preset")
        appearance = _PRESET_APPEARANCES.get(preset) if isinstance(preset, str) else None
        if type(settings_hwnd) is not int or settings_hwnd <= 0:
            settings_hwnd = None
        window = QwenRulesWindow(
            self._parent, self._root, self._language, self._selected_rule, self._select_rule,
            restore_parent_grab=False, appearance=appearance,
        )
        self._rules_window = window

        def closed(event):
            if event.widget is window:
                self._rules_window = None
                if settings_hwnd:
                    try:
                        raise_settings_window(settings_hwnd)
                    except (Win32Error, OverflowError):
                        pass
                reply({"rule_name": self._rule_name()})

        window.bind("<Destroy>", closed, add="+")
        window.attributes("-topmost", True)
        window.lift()
        window.focus_force()
        if settings_hwnd:
            try:
                place_window_beside(window, settings_hwnd)
            except (Win32Error, OverflowError):
                # Settings can close while Tk is creating the rules dialog.
                pass

    def _select_rule(self, rule):
        self._selected_rule = rule
