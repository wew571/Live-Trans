"""Select a live translation preset or edit its component settings."""

from dataclasses import dataclass, replace
from math import cos, tau
from time import monotonic
from tkinter import StringVar
from tkinter.colorchooser import askcolor

import customtkinter as ctk
from PIL import Image

from core.config import ASSETS_DIR
from core.live_models import LiveModelError, check_qwen_cuda_worker
from core.ocr_settings import (
    DEFAULT_JA_MODEL,
    DEFAULT_TRANSLATION_COLOR,
    FLOW_DIRECT,
    FLOW_NLLB_VINAI,
    FLOW_PIVOT,
    HISTORY_TURNS_MAX,
    POST_TRANSLATION_DELAY_MAX,
    valid_post_translation_delay,
    INTERFACE_ADVANCED,
    INTERFACE_MINIMAL,
    OCR_SIZES,
    PRESETS,
    QWEN_BACKEND_CUDA_PYTHON,
    QWEN_BACKEND_LABELS,
    QWEN_BACKEND_VULKAN,
    QWEN_CONTEXT_MAX,
    QWEN_CONTEXT_MIN,
    QWEN_MODEL_LABELS,
    valid_history_turns,
)
from core.setup_models import model_items
from frontend.preset_slider import PresetSlider
from frontend.qwen_rules_window import QwenRulesWindow


_OCR_OPTIONS = {f"PP-OCRv6 {size.title()}": size for size in OCR_SIZES}
_JA_OPTIONS = {"Mitsua ElanMT": DEFAULT_JA_MODEL, "NLLB-200 Distilled 600M": "nllb"}
_PRESET_NAMES = tuple(PRESETS)
_LOCKED_FLOW_PRESETS = ("Flash", "Medium")


@dataclass(frozen=True)
class _PresetAppearance:
    ink: str
    highlight: str
    surface: str
    border: str
    control: str
    hover: str
    selected: str


_STANDARD = _PresetAppearance(
    "#e9ecfa", "#e9ecfa", "#242438", "#3a3a52", "#216699", "#144870", "#334a73",
)
_PRESET_APPEARANCES = {
    "Ultra": _PresetAppearance("#d3c7ff", "#eee3ff", "#272542", "#7774cc", "#373c73", "#474e8e", "#464b92"),
    "Extreme": _PresetAppearance("#b5f4c0", "#e4ffce", "#20382e", "#509668", "#24523e", "#347050", "#336c4d"),
    "Ultimate": _PresetAppearance("#ffcf9f", "#fff0c5", "#3d282a", "#bb713d", "#653822", "#854a2b", "#8b481f"),
}


def _mix_colors(start, end, amount):
    channels = [(int(start[i:i + 2], 16), int(end[i:i + 2], 16)) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(low + (high - low) * amount):02x}" for low, high in channels)


_LABELS = {
    "VI": {
        "title": "Cài đặt Live Trans", "custom": "Tùy chỉnh", "advanced": "Nâng cao",
        "interface": "Giao diện", "minimal_interface": "Tối giản", "advanced_interface": "Nâng cao",
        "translation_color": "Màu chữ bản dịch", "choose_color": "Chọn màu", "default_color": "Mặc định",
        "crop_count": "Số vùng dịch", "history_turns": "Lượt thoại trong ngữ cảnh (0 = Tắt)",
        "post_translation_delay": "Thời gian chờ sau khi hiện bản dịch (ms)",
        "delay_hint": "0–5.000 ms; 0 = không chờ.",
        "invalid_delay": "Nhập thời gian chờ nguyên từ 0 đến 5.000 ms.",
        "invalid_history": f"Nhập số lượt thoại nguyên từ 0 đến {HISTORY_TURNS_MAX}.",
        "det": "Dò vùng chữ (det)", "rec": "Đọc nội dung (rec)", "flow": "Luồng dịch",
        "ja_model": "Mô hình Nhật → Anh", "context": "Context {model} (token, áp dụng sau khi khởi động lại)",
        "qwen_backend": "Backend Qwen",
        "backend_default": "{name} (mặc định)",
        "backend_optional": "{name} (bổ sung)",
        "backend_hint": "CUDA cần GPU NVIDIA, PyTorch CUDA 13 và runtime bổ sung trong Thiết lập.",
        "pivot": "Nhật → Anh → Việt", "direct": "Nhật → Việt · Anh → Việt",
        "nllb-vinai": "Nhật → Việt · Anh → Việt (NLLB + VinAI)",
        "english_direct": "Anh → Việt",
        "guide": "Áp dụng ở lần Bắt đầu tiếp theo.",
        "save": "Lưu", "missing": "Đóng Cài đặt và mở Thiết lập để tải:\n{models}",
        "invalid_context": "Nhập context Qwen từ 1.000 đến 25.000 token.",
        "error": "Chưa lưu được cài đặt.",
        "active_rule": "Quy tắc chung · {name}", "no_rule": "Không có", "manage_rules": "Quản lý quy tắc",
        "shared_rule_hint": "Áp dụng cho tất cả vùng khi dùng Qwen. Bấm Lưu để áp dụng.",
    },
    "EN": {
        "title": "Live Trans settings", "custom": "Custom", "advanced": "Advanced",
        "interface": "Interface", "minimal_interface": "Minimal", "advanced_interface": "Advanced",
        "translation_color": "Translation text color", "choose_color": "Choose color", "default_color": "Default",
        "crop_count": "Translation regions", "history_turns": "Dialogue turns in context (0 = Off)",
        "post_translation_delay": "Wait after showing translation (ms)",
        "delay_hint": "0–5,000 ms; 0 = no wait.",
        "invalid_delay": "Enter a whole number from 0 to 5,000 ms for the delay.",
        "invalid_history": f"Enter a whole number from 0 to {HISTORY_TURNS_MAX} for dialogue turns.",
        "det": "Find text regions (det)", "rec": "Read text (rec)", "flow": "Translation flow",
        "ja_model": "Japanese → English model", "context": "{model} context (tokens, applies after restart)",
        "qwen_backend": "Qwen backend",
        "backend_default": "{name} (default)",
        "backend_optional": "{name} (optional)",
        "backend_hint": "CUDA requires an NVIDIA GPU, PyTorch CUDA 13 and the optional runtime in Setup.",
        "pivot": "Japanese → English → Vietnamese", "direct": "Japanese → Vietnamese · English → Vietnamese",
        "nllb-vinai": "Japanese → Vietnamese · English → Vietnamese (NLLB + VinAI)",
        "english_direct": "English → Vietnamese",
        "guide": "Applies the next time you press Start.",
        "save": "Save", "missing": "Close Settings and open Setup to install:\n{models}",
        "invalid_context": "Enter a Qwen context from 1,000 to 25,000 tokens.",
        "error": "Settings could not be saved.",
        "active_rule": "Shared rule · {name}", "no_rule": "None", "manage_rules": "Manage rules",
        "shared_rule_hint": "Applies to all regions when using Qwen. Select Save to apply.",
    },
}


def _label_for(options, value):
    return next(label for label, option in options.items() if option == value)


def _backend_options(texts):
    return {
        texts[designation].format(name=QWEN_BACKEND_LABELS[backend]): backend
        for backend, designation in (
            (QWEN_BACKEND_VULKAN, "backend_default"),
            (QWEN_BACKEND_CUDA_PYTHON, "backend_optional"),
        )
    }


def _routing_preset_index(flow, ja_model, qwen_model):
    for index, preset in enumerate(PRESETS.values()):
        if preset.flow != flow:
            continue
        if flow == FLOW_NLLB_VINAI or (preset.qwen_model if flow == FLOW_DIRECT else preset.ja_model) == (
            qwen_model if flow == FLOW_DIRECT else ja_model
        ):
            return index
    return 0


def _initial_preset_index(settings):
    if settings.preset:
        return _PRESET_NAMES.index(settings.preset)
    return _routing_preset_index(settings.flow, settings.ja_model, settings.qwen_model)


def selection_status(root, settings, texts, *, original, saving=False):
    """Return the same validation message and severity for either settings window."""
    if settings is None:
        return texts["invalid_context"], "#f9e2af"
    check_pipeline = settings == original or not settings.same_pipeline_as(original)
    if check_pipeline:
        try:
            missing = [item.label for item in model_items(root, settings) if not item.is_ready()]
        except Exception as error:
            return f"{texts['error']}\n{error}", "#f38ba8"
        if missing:
            models = "\n".join(f"• {label}" for label in missing)
            return texts["missing"].format(models=models), "#f9e2af"
    if saving and check_pipeline and settings.flow == FLOW_DIRECT and settings.qwen_backend == QWEN_BACKEND_CUDA_PYTHON:
        try:
            check_qwen_cuda_worker()
        except LiveModelError as error:
            return f"{texts['error']}\n{error}", "#f38ba8"
    return "", "#f9e2af"


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
        self._locked_flow = settings.preset in _LOCKED_FLOW_PRESETS
        self._rules_window = None
        self._appearance = None
        self._appearance_job = None
        self._appearance_started = 0.0
        self._ultimate_sky = None
        self._field_labels = {}
        self._flow_options = {self._texts[flow]: flow for flow in (FLOW_PIVOT, FLOW_NLLB_VINAI, FLOW_DIRECT)}
        self._backend_options = _backend_options(self._texts)

        self.title(self._texts["title"])
        self.transient(parent)
        self.resizable(False, False)
        self.configure(fg_color="#181825")
        self.bind("<Escape>", lambda _event: self.destroy())

        content = ctk.CTkFrame(self, fg_color="transparent", width=394)
        content.pack(padx=20, pady=16, fill="both", expand=True)
        self.title_label = ctk.CTkLabel(content, text=self._texts["title"], font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(anchor="w")

        self.levels = ctk.CTkFrame(content, fg_color=_STANDARD.surface, border_width=1, corner_radius=8)
        self.levels.pack(fill="x", pady=(12, 8))
        self.slider = PresetSlider(
            self.levels, stop_count=len(PRESETS), initial_index=_initial_preset_index(settings),
            command=self._choose_preset,
            animation_enabled=settings.settings_interface == INTERFACE_ADVANCED and settings.animation_enabled,
        )
        self.slider.pack(fill="x", padx=10, pady=(8, 0))
        labels = ctk.CTkFrame(self.levels, fg_color="transparent")
        labels.pack(fill="x", padx=8)
        self.preset_buttons = []
        for index, name in enumerate(_PRESET_NAMES):
            labels.grid_columnconfigure(index, weight=1, uniform="presets")
            button = ctk.CTkButton(
                labels, text=name, width=54, height=28, fg_color="transparent", hover_color="#313244",
                text_color="#cdd6f4", command=lambda index=index: self._choose_preset(index),
            )
            button.grid(row=0, column=index, sticky="ew", padx=1)
            self.preset_buttons.append(button)

        self.level_divider = ctk.CTkFrame(self.levels, height=1, fg_color=_STANDARD.border)
        self.level_divider.pack(fill="x", padx=14, pady=(8, 3))
        self.preset_name = ctk.CTkLabel(self.levels, height=20, font=ctk.CTkFont(size=15, weight="bold"), text="")
        self.preset_name.pack(anchor="w", padx=14)
        self.summary = ctk.CTkLabel(self.levels, text="", text_color="#bac2de", justify="left", wraplength=365)
        self.summary.pack(anchor="w", padx=14, pady=(0, 8))

        interface_row = ctk.CTkFrame(content, fg_color="transparent")
        interface_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(interface_row, text=self._texts["interface"]).pack(side="left")
        self._interface_options = {
            self._texts["minimal_interface"]: INTERFACE_MINIMAL,
            self._texts["advanced_interface"]: INTERFACE_ADVANCED,
        }
        self.interface_menu = ctk.CTkOptionMenu(
            interface_row, values=list(self._interface_options), width=136, command=self._update_selection,
        )
        self.interface_menu.set(_label_for(self._interface_options, settings.settings_interface))
        self.interface_menu.pack(side="right")

        color_row = ctk.CTkFrame(content, fg_color="transparent")
        color_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(color_row, text=self._texts["translation_color"]).pack(side="left")
        self.translation_color_reset = ctk.CTkButton(
            color_row, width=68, text=self._texts["default_color"], command=self._reset_translation_color,
        )
        self.translation_color_reset.pack(side="right")
        self.translation_color_button = ctk.CTkButton(
            color_row, width=76, text=self._texts["choose_color"], command=self._choose_translation_color,
        )
        self.translation_color_button.pack(side="right", padx=6)
        self._translation_color_swatch = ctk.CTkLabel(color_row, width=20, height=20, text="", corner_radius=4)
        self._translation_color_swatch.pack(side="right")
        self._set_translation_color(settings.translation_color)

        crop_row, self.crop_count_menu = self._numeric_row(content, "crop_count", settings.crop_count, (1, 2, 3))
        crop_row.pack(fill="x", pady=(0, 8))
        delay_row = ctk.CTkFrame(content, fg_color="transparent")
        delay_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(delay_row, text=self._texts["post_translation_delay"], wraplength=250).pack(side="left")
        self.delay_entry = ctk.CTkEntry(delay_row, width=100)
        self.delay_entry.insert(0, str(settings.post_translation_delay_ms))
        self.delay_entry.pack(side="right")
        ctk.CTkLabel(content, text=self._texts["delay_hint"], text_color="#bac2de").pack(anchor="w")
        self.history_row = ctk.CTkFrame(content, fg_color="transparent")
        ctk.CTkLabel(self.history_row, text=self._texts["history_turns"]).pack(side="left")
        self.history_entry = ctk.CTkEntry(self.history_row, width=136)
        self.history_entry.insert(0, str(settings.history_turns))
        self.history_entry.pack(side="right")

        self.animation_switch = ctk.CTkSwitch(content, text="Animation", command=self._update_selection)
        if settings.animation_enabled:
            self.animation_switch.select()
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
        self.backend_menu = self._model_menu(
            self.context_frame, "qwen_backend", settings.qwen_backend, self._backend_options,
        )
        ctk.CTkLabel(
            self.context_frame, text=self._texts["backend_hint"], text_color="#bac2de",
            anchor="w", justify="left", wraplength=380,
        ).pack(fill="x", pady=(2, 0))
        self.active_rule_label = ctk.CTkLabel(self.context_frame, text="", anchor="w", wraplength=380)
        self.active_rule_label.pack(fill="x", pady=(8, 2))
        ctk.CTkLabel(
            self.context_frame, text=self._texts["shared_rule_hint"], text_color="#bac2de",
            anchor="w", justify="left", wraplength=380,
        ).pack(fill="x", pady=(0, 4))
        self.manage_rules_button = ctk.CTkButton(self.context_frame, text=self._texts["manage_rules"], command=self._manage_rules)
        self.manage_rules_button.pack(fill="x")
        self._update_rule_label()

        self.guide = ctk.CTkLabel(content, text=self._texts["guide"], text_color="#bac2de")
        self.guide.pack(anchor="w", pady=(8, 2))
        self.status = ctk.CTkLabel(content, text="", anchor="w", justify="left", wraplength=385)
        self.save_button = ctk.CTkButton(content, text=self._texts["save"], command=self._save)
        self.save_button.pack(fill="x")
        self.context_entry.bind("<KeyRelease>", self._update_selection)
        self.history_entry.bind("<KeyRelease>", self._update_selection)
        self.delay_entry.bind("<KeyRelease>", self._update_selection)
        self.bind("<Map>", self._map_appearance, add="+")
        self.bind("<Unmap>", self._unmap_appearance, add="+")
        self._update_selection()

        self.update_idletasks()
        self.geometry(f"+{parent.winfo_x() + 35}+{parent.winfo_y() + 60}")
        self.grab_set()
        self.focus_set()

    def _model_menu(self, parent, role, value, options):
        label = ctk.CTkLabel(parent, text=self._texts[role])
        label.pack(anchor="w", pady=(8, 2))
        self._field_labels[role] = label
        menu = ctk.CTkOptionMenu(parent, values=list(options), command=self._update_selection, width=380)
        menu.set(_label_for(options, value))
        menu.pack(fill="x")
        return menu

    def _numeric_row(self, parent, role, value, choices):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(row, text=self._texts[role]).pack(side="left")
        labels = [str(choice) for choice in choices]
        menu = ctk.CTkOptionMenu(row, values=labels, width=136, command=self._update_selection)
        menu.set(labels[choices.index(value)])
        menu.pack(side="right")
        return row, menu

    def destroy(self):
        self._cancel_appearance()
        super().destroy()

    def _map_appearance(self, event):
        if event.widget is self and self._appearance not in (None, _STANDARD):
            self._schedule_appearance()

    def _unmap_appearance(self, event):
        if event.widget is self:
            self._cancel_appearance()

    def _cancel_appearance(self):
        if self._appearance_job is not None:
            self.after_cancel(self._appearance_job)
            self._appearance_job = None

    def _schedule_appearance(self):
        if self._appearance_job is None:
            self._appearance_job = self.after(85, self._animate_appearance)

    def _animate_appearance(self):
        self._appearance_job = None
        theme = self._appearance
        if not self.winfo_viewable() or theme is _STANDARD:
            return
        elapsed = monotonic() - self._appearance_started
        phase = elapsed * tau / 3
        self._paint_appearance_text(theme, phase)
        self._schedule_appearance()

    def _paint_appearance_text(self, theme, phase=0.0):
        headings = (self.title_label, self.preset_name)
        fields = (self.advanced_button, *self._field_labels.values(), self.context_label)
        actions = (self.manage_rules_button, self.save_button)
        if theme is _STANDARD:
            groups = ((headings + fields + actions + (self.active_rule_label,), theme.ink), ((self.summary,), "#bac2de"))
        else:
            groups = tuple(
                (widgets, _mix_colors(theme.ink, theme.highlight, (1 - cos(phase + offset)) / 2))
                for widgets, offset in (
                    (headings, 0.8), ((self.summary, self.active_rule_label), 1.8),
                    (fields, 0.0), (actions, 0.4),
                )
            )
        for widgets, color in groups:
            for widget in widgets:
                if widget.cget("text_color") != color:
                    widget.configure(text_color=color)

    def _show_ultimate_sky(self, visible):
        if not visible:
            if self._ultimate_sky is not None:
                self._ultimate_sky.place_forget()
            return
        if self._ultimate_sky is None:
            with Image.open(ASSETS_DIR / "ultimate-preset-sky.png") as sky:
                image = ctk.CTkImage(sky.copy(), size=(166, 68))
            self._ultimate_sky = ctk.CTkLabel(self.levels, text="", image=image)
        self._ultimate_sky.place(relx=1, rely=1, x=-3, y=-3, anchor="se")
        self._ultimate_sky.lower(self.preset_name)
        self.summary.lift()

    def _apply_appearance(self, index, preset):
        name = _PRESET_NAMES[index]
        advanced = self._interface() == INTERFACE_ADVANCED
        if advanced:
            if not self.animation_switch.winfo_manager():
                self.animation_switch.pack(anchor="w", pady=(0, 8), before=self.advanced_button)
        else:
            self.animation_switch.pack_forget()
        animated = advanced and bool(self.animation_switch.get())
        theme = _PRESET_APPEARANCES.get(name, _STANDARD) if animated else _STANDARD
        self.slider.set_animation_enabled(animated)
        self.levels.configure(border_width=int(animated))
        if animated:
            if not self.level_divider.winfo_manager():
                before = self.preset_name if self.preset_name.winfo_manager() else self.summary
                self.level_divider.pack(fill="x", padx=14, pady=(8, 3), before=before)
        else:
            self.level_divider.pack_forget()
        if animated or not preset:
            if not self.preset_name.winfo_manager():
                self.preset_name.pack(anchor="w", padx=14, before=self.summary)
        else:
            self.preset_name.pack_forget()
        if theme is not self._appearance:
            self._appearance = theme
            self._cancel_appearance()
            self.levels.configure(fg_color=theme.surface, border_color=theme.border)
            self.slider.set_surface_color(theme.surface)
            self.level_divider.configure(fg_color=theme.border)
            self._show_ultimate_sky(theme is _PRESET_APPEARANCES["Ultimate"])
            for menu in (self.det_menu, self.rec_menu, self.flow_menu, self.ja_menu):
                menu.configure(fg_color=theme.control, button_color=theme.hover, button_hover_color=theme.control)
            self.context_entry.configure(border_color=theme.border)
            for button in (self.manage_rules_button, self.save_button):
                button.configure(fg_color=theme.control, hover_color=theme.hover)
            self._paint_appearance_text(theme)
            if theme is not _STANDARD:
                self._appearance_started = monotonic()
                if self.winfo_viewable():
                    self._schedule_appearance()
        for name, button in zip(_PRESET_NAMES, self.preset_buttons):
            button.configure(fg_color=theme.selected if name == preset else "transparent")

    def _choose_preset(self, index):
        index = round(index)
        name = _PRESET_NAMES[index]
        preset = PRESETS[name]
        self.slider.set(index)
        self._locked_flow = name in _LOCKED_FLOW_PRESETS
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

    def _history_value(self):
        try:
            value = int(self.history_entry.get())
        except ValueError:
            return None
        return value if valid_history_turns(value) else None

    def _choose_translation_color(self):
        _, color = askcolor(
            color=self._translation_color or DEFAULT_TRANSLATION_COLOR,
            parent=self, title=self._texts["translation_color"],
        )
        if color is not None:
            self._set_translation_color(color)
            self._update_selection()

    def _set_translation_color(self, color):
        self._translation_color = color
        self._translation_color_swatch.configure(fg_color=color or DEFAULT_TRANSLATION_COLOR)
        self.translation_color_reset.configure(state="normal" if color else "disabled")

    def _reset_translation_color(self):
        self._set_translation_color(None)
        self._update_selection()

    def _selection(self):
        flow = self._flow_options[self.flow_menu.get()]
        try:
            delay = int(self.delay_entry.get())
        except ValueError:
            return None
        if not valid_post_translation_delay(delay):
            return None
        history = self._history_value()
        if history is None:
            if flow == FLOW_DIRECT:
                return None
            history = self._original.history_turns
        try:
            context = int(self.context_text.get())
        except ValueError:
            context = 0
        if not QWEN_CONTEXT_MIN <= context <= QWEN_CONTEXT_MAX:
            if flow == FLOW_DIRECT:
                return None
            context = self._original.qwen_context
        return replace(
            self._original, det=_OCR_OPTIONS[self.det_menu.get()], rec=_OCR_OPTIONS[self.rec_menu.get()],
            ja_model=_JA_OPTIONS[self.ja_menu.get()], flow=flow, qwen_context=context,
            qwen_rule=self._selected_rule, qwen_model=self._qwen_model,
            qwen_backend=self._backend_options[self.backend_menu.get()],
            animation_enabled=bool(self.animation_switch.get()),
            settings_interface=self._interface(),
            crop_count=int(self.crop_count_menu.get()),
            history_turns=history,
            post_translation_delay_ms=delay,
            translation_color=self._translation_color,
        )

    def _interface(self):
        return self._interface_options[self.interface_menu.get()]

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
        if direct and not self.history_row.winfo_manager():
            self.history_row.pack(fill="x", pady=(0, 8), before=self.advanced_button)
        elif not direct:
            self.history_row.pack_forget()
        self.context_label.configure(text=self._texts["context"].format(model=QWEN_MODEL_LABELS[self._qwen_model]))
        for frame, visible in ((self.context_frame, direct), (self.pivot_frame, flow == FLOW_PIVOT)):
            if visible:
                frame.pack(fill="x")
            else:
                frame.pack_forget()
        settings = self._selection()
        preset = settings.preset if settings else None
        self.preset_name.configure(text=preset or self._texts["custom"])
        index = _routing_preset_index(flow, _JA_OPTIONS[self.ja_menu.get()], self._qwen_model)
        if preset:
            index = _PRESET_NAMES.index(preset)
            self._locked_flow = preset in _LOCKED_FLOW_PRESETS
        self.slider.set(index)
        self._apply_appearance(index, preset)
        self.flow_menu.configure(state="disabled" if self._locked_flow else "normal")
        ocr = f"PP-OCRv6 · {self.det_menu.get().split()[-1]} det · {self.rec_menu.get().split()[-1]} rec"
        route = self._texts[FLOW_DIRECT]
        if direct:
            translator = QWEN_MODEL_LABELS[self._qwen_model]
        elif flow == FLOW_NLLB_VINAI:
            translator = "NLLB Custom ONNX · VinAI v2"
        else:
            translator = f"{self.ja_menu.get()} → VinAI v2"
            route = f"{self._texts[flow]} · {self._texts['english_direct']}"
        self.summary.configure(text=f"{ocr}\n{translator}\n{route}")
        self._check_selection(settings)

    def _check_selection(self, settings, *, saving=False):
        if settings is None:
            try:
                delay = int(self.delay_entry.get())
            except ValueError:
                delay = -1
            if not valid_post_translation_delay(delay):
                self._show_status(self._texts["invalid_delay"], "#f38ba8")
                return False
        if settings is None and self._history_value() is None:
            self._show_status(self._texts["invalid_history"], "#f38ba8")
            return False
        message, color = selection_status(
            self._root_path, settings, self._texts, original=self._original, saving=saving,
        )
        self._show_status(message, color)
        return not message

    def _save(self):
        settings = self._selection()
        if not self._check_selection(settings, saving=True):
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
