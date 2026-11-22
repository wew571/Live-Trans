"""Select a live translation preset or edit its component settings."""

from tkinter import StringVar

import customtkinter as ctk

from core.ocr_settings import (
    DEFAULT_JA_MODEL,
    FLOW_DIRECT,
    FLOW_PIVOT,
    OCR_SIZES,
    PRESETS,
    QWEN_CONTEXT_MAX,
    QWEN_CONTEXT_MIN,
    QWEN_MODEL_LABELS,
    OCRSettings,
)
from core.setup_models import model_items
from frontend.qwen_rules_window import QwenRulesWindow


_OCR_OPTIONS = {f"PP-OCRv6 {size.title()}": size for size in OCR_SIZES}
_JA_OPTIONS = {"Mitsua ElanMT": DEFAULT_JA_MODEL, "NLLB-200 Distilled 600M": "nllb"}
_PRESET_NAMES = tuple(PRESETS)
_LOCKED_PIVOT_PRESETS = ("Flash", "Medium")
_LABELS = {
    "VI": {
        "title": "Cài đặt Live Trans", "custom": "Tùy chỉnh", "advanced": "Nâng cao",
        "det": "Dò vùng chữ (det)", "rec": "Đọc nội dung (rec)", "flow": "Luồng dịch",
        "ja_model": "Mô hình Nhật → Anh", "context": "Context {model} (token)",
        "pivot": "Nhật → Anh → Việt", "direct": "Nhật → Việt · Anh → Việt",
        "guide": "Áp dụng ở lần Bắt đầu tiếp theo.",
        "save": "Lưu", "missing": "Đóng Cài đặt và mở Thiết lập để tải:\n{models}",
        "invalid_context": "Nhập context Qwen từ 1.000 đến 25.000 token.",
        "error": "Chưa lưu được cài đặt.",
        "active_rule": "Quy tắc đang chọn: {name}", "no_rule": "Không có", "manage_rules": "Quản lý quy tắc",
    },
    "EN": {
        "title": "Live Trans settings", "custom": "Custom", "advanced": "Advanced",
        "det": "Find text regions (det)", "rec": "Read text (rec)", "flow": "Translation flow",
        "ja_model": "Japanese → English model", "context": "{model} context (tokens)",
        "pivot": "Japanese → English → Vietnamese", "direct": "Japanese → Vietnamese · English → Vietnamese",
        "guide": "Applies the next time you press Start.",
        "save": "Save", "missing": "Close Settings and open Setup to install:\n{models}",
        "invalid_context": "Enter a Qwen context from 1,000 to 25,000 tokens.",
        "error": "Settings could not be saved.",
        "active_rule": "Selected rule: {name}", "no_rule": "None", "manage_rules": "Manage rules",
    },
}


def _label_for(options, value):
    return next(label for label, option in options.items() if option == value)


class OCRSettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, root, settings, language, on_save):
        super().__init__(parent)
        self._root_path = root
        self._original = settings
        self._on_save = on_save
        self._texts = _LABELS[language]
        self._language = language
        self._selected_rule = settings.qwen_rule
        self._qwen_model = settings.qwen_model
        self._locked_flow = settings.preset in _LOCKED_PIVOT_PRESETS
        self._rules_window = None
        self._flow_options = {self._texts[flow]: flow for flow in (FLOW_PIVOT, FLOW_DIRECT)}

        self.title(self._texts["title"])
        self.transient(parent)
        self.resizable(False, False)
        self.configure(fg_color="#181825")
        self.bind("<Escape>", lambda _event: self.destroy())

        content = ctk.CTkFrame(self, fg_color="transparent", width=394)
        content.pack(padx=20, pady=16, fill="both", expand=True)
        ctk.CTkLabel(content, text=self._texts["title"], font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")

        levels = ctk.CTkFrame(content, fg_color="#242438", corner_radius=8)
        levels.pack(fill="x", pady=(12, 8))
        self.preset_name = ctk.CTkLabel(levels, font=ctk.CTkFont(size=15, weight="bold"), text="")
        self.preset_name.pack(anchor="w", padx=14, pady=(10, 0))
        self.summary = ctk.CTkLabel(levels, text="", text_color="#bac2de", justify="left", wraplength=365)
        self.summary.pack(anchor="w", padx=14, pady=(0, 8))
        self.slider = ctk.CTkSlider(
            levels, from_=0, to=len(PRESETS) - 1, number_of_steps=len(PRESETS) - 1,
            width=359, command=self._choose_preset,
        )
        initial_index = _PRESET_NAMES.index(settings.preset) if settings.preset else (
            len(PRESETS) - 1 if settings.flow == FLOW_DIRECT else 0
        )
        self.slider.set(initial_index)
        self.slider.pack(fill="x", padx=14)
        labels = ctk.CTkFrame(levels, fg_color="transparent")
        labels.pack(fill="x", padx=8, pady=(1, 6))
        for index, name in enumerate(_PRESET_NAMES):
            labels.grid_columnconfigure(index, weight=1, uniform="presets")
            ctk.CTkButton(
                labels, text=name, width=54, height=22, fg_color="transparent", hover_color="#313244",
                text_color="#cdd6f4", command=lambda index=index: self._choose_preset(index),
            ).grid(row=0, column=index, sticky="ew", padx=1)

        self.advanced_button = ctk.CTkButton(
            content, text=f"▸ {self._texts['advanced']}", anchor="w", fg_color="#313244",
            hover_color="#45475a", command=self._toggle_advanced,
        )
        self.advanced_button.pack(fill="x")
        self.advanced_frame = ctk.CTkFrame(content, fg_color="transparent")
        self.det_menu = self._model_menu(self.advanced_frame, "det", settings.det, _OCR_OPTIONS)
        self.rec_menu = self._model_menu(self.advanced_frame, "rec", settings.rec, _OCR_OPTIONS)
        self.flow_menu = self._model_menu(self.advanced_frame, "flow", settings.flow, self._flow_options)
        self.pivot_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        self.ja_menu = self._model_menu(self.pivot_frame, "ja_model", settings.ja_model, _JA_OPTIONS)
        self.context_frame = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
        self.context_label = ctk.CTkLabel(self.context_frame, text="")
        self.context_label.pack(anchor="w", pady=(8, 2))
        self.context_text = StringVar(self, str(settings.qwen_context))
        self.context_entry = ctk.CTkEntry(self.context_frame, textvariable=self.context_text)
        self.context_entry.pack(fill="x")
        self.active_rule_label = ctk.CTkLabel(self.context_frame, text="", anchor="w", wraplength=380)
        self.active_rule_label.pack(fill="x", pady=(8, 2))
        self.manage_rules_button = ctk.CTkButton(self.context_frame, text=self._texts["manage_rules"], command=self._manage_rules)
        self.manage_rules_button.pack(fill="x")
        self._update_rule_label()

        self.guide = ctk.CTkLabel(content, text=self._texts["guide"], text_color="#bac2de")
        self.guide.pack(anchor="w", pady=(8, 2))
        self.status = ctk.CTkLabel(content, text="", anchor="w", justify="left", wraplength=385)
        self.save_button = ctk.CTkButton(content, text=self._texts["save"], command=self._save)
        self.save_button.pack(fill="x")
        self.context_entry.bind("<KeyRelease>", self._update_selection)
        self._update_selection()

        self.update_idletasks()
        self.geometry(f"+{parent.winfo_x() + 35}+{parent.winfo_y() + 60}")
        self.grab_set()
        self.focus_set()

    def _model_menu(self, parent, role, value, options):
        ctk.CTkLabel(parent, text=self._texts[role]).pack(anchor="w", pady=(8, 2))
        menu = ctk.CTkOptionMenu(parent, values=list(options), command=self._update_selection, width=380)
        menu.set(_label_for(options, value))
        menu.pack(fill="x")
        return menu

    def _choose_preset(self, index):
        index = round(index)
        name = _PRESET_NAMES[index]
        preset = PRESETS[name]
        self.slider.set(index)
        self._locked_flow = name in _LOCKED_PIVOT_PRESETS
        if preset.flow == FLOW_DIRECT:
            self._qwen_model = preset.qwen_model
        self.det_menu.set(_label_for(_OCR_OPTIONS, preset.det))
        self.rec_menu.set(_label_for(_OCR_OPTIONS, preset.rec))
        self.flow_menu.set(_label_for(self._flow_options, preset.flow))
        if preset.flow == FLOW_PIVOT:
            self.ja_menu.set(_label_for(_JA_OPTIONS, preset.ja_model))
        self._update_selection()

    def _toggle_advanced(self):
        if self.advanced_frame.winfo_manager():
            self.advanced_frame.pack_forget()
            arrow = "▸"
        else:
            self.advanced_frame.pack(fill="x", before=self.guide)
            arrow = "▾"
        self.advanced_button.configure(text=f"{arrow} {self._texts['advanced']}")

    def _selection(self):
        flow = self._flow_options[self.flow_menu.get()]
        try:
            context = int(self.context_text.get())
        except ValueError:
            context = 0
        if not QWEN_CONTEXT_MIN <= context <= QWEN_CONTEXT_MAX:
            if flow == FLOW_DIRECT:
                return None
            context = self._original.qwen_context
        return OCRSettings(
            _OCR_OPTIONS[self.det_menu.get()], _OCR_OPTIONS[self.rec_menu.get()],
            _JA_OPTIONS[self.ja_menu.get()], flow, context, self._selected_rule, self._qwen_model,
        )

    def _update_rule_label(self):
        name = self._selected_rule.name if self._selected_rule else self._texts["no_rule"]
        self.active_rule_label.configure(text=self._texts["active_rule"].format(name=name))

    def _select_rule(self, rule):
        self._selected_rule = rule
        self._update_rule_label()
        self._update_selection()

    def _manage_rules(self):
        if self._rules_window is not None and self._rules_window.winfo_exists():
            self._rules_window.focus_set()
            return
        self._rules_window = QwenRulesWindow(
            self, self._root_path, self._language, self._selected_rule, self._select_rule,
        )

    def _show_status(self, message="", color="#f9e2af"):
        self.status.configure(text=message, text_color=color)
        if message:
            self.status.pack(fill="x", before=self.save_button, pady=(0, 8))
        else:
            self.status.pack_forget()

    def _update_selection(self, _event=None):
        flow = self._flow_options[self.flow_menu.get()]
        direct = flow == FLOW_DIRECT
        self.context_label.configure(text=self._texts["context"].format(model=QWEN_MODEL_LABELS[self._qwen_model]))
        active, inactive = (self.context_frame, self.pivot_frame) if direct else (
            self.pivot_frame, self.context_frame,
        )
        inactive.pack_forget()
        active.pack(fill="x")
        settings = self._selection()
        preset = settings.preset if settings else None
        self.preset_name.configure(text=preset or self._texts["custom"])
        if preset:
            self.slider.set(_PRESET_NAMES.index(preset))
            self._locked_flow = preset in _LOCKED_PIVOT_PRESETS
        self.flow_menu.configure(state="disabled" if self._locked_flow else "normal")
        ocr = f"OCR {self.det_menu.get().split()[-1]} det · {self.rec_menu.get().split()[-1]} rec"
        translator = QWEN_MODEL_LABELS[self._qwen_model] if direct else f"{self.ja_menu.get()} · VinAI v2"
        self.summary.configure(text=f"{ocr}\n{translator}")
        self._check_selection(settings)

    def _check_selection(self, settings):
        if settings is None:
            self._show_status(self._texts["invalid_context"])
            return False
        try:
            missing = [item.label for item in model_items(self._root_path, settings) if not item.is_ready()]
        except Exception as error:
            self._show_status(f"{self._texts['error']}\n{error}", "#f38ba8")
            return False
        if missing:
            models = "\n".join(f"• {label}" for label in missing)
            self._show_status(self._texts["missing"].format(models=models))
            return False
        self._show_status()
        return True

    def _save(self):
        settings = self._selection()
        if not self._check_selection(settings):
            return
        if settings == self._original:
            self.destroy()
            return
        try:
            settings.save(self._root_path)
        except OSError as error:
            self._show_status(f"{self._texts['error']}\n{error}", "#f38ba8")
            return
        self.destroy()
        self._on_save(settings)

