"""Manage reusable Qwen rules without applying Settings immediately."""

from tkinter import StringVar

import customtkinter as ctk

from core.qwen_rules import load_rule_slots, save_rule_slot
from frontend.windows_shell import set_dark_title_bar


_LABELS = {
    "VI": {"title": "Quy tắc dịch Qwen", "new": "＋ Tạo mới", "none": "Không dùng quy tắc",
           "slot": "Slot", "name": "Tên slot", "content": "Nội dung quy tắc",
           "save": "Lưu quy tắc", "hint": "Với crop tên nhân vật, đặt “Vai trò: Tên người nói” ở dòng đầu. "
           "Lưu slot rồi chọn slot đó trên màn chính.",
           "saved": "Đã lưu slot “{name}”.", "error": "Không lưu được quy tắc."},
    "EN": {"title": "Manage rules", "new": "＋ New rule", "none": "No rule",
           "slot": "Slot", "name": "Rule name", "content": "Rule content",
           "save": "Save rule", "hint": "For a character name crop, put “Role: Speaker” on the first line. "
           "Save the slot and select it on the dashboard.",
           "saved": "Saved slot “{name}”.", "error": "Could not save the rule."},
}


class QwenRulesWindow(ctk.CTkToplevel):
    def __init__(self, parent, root, language, selected, on_select, *, restore_parent_grab=True, appearance=None):
        # The WebView will restore its own focus. Avoid CustomTkinter's deferred
        # Windows title-bar redraw, which can reenter Tk while this editor closes.
        self._deactivate_windows_window_header_manipulation = not restore_parent_grab
        super().__init__(parent)
        self._parent = parent
        self._restore_parent_grab = restore_parent_grab
        self._root_path = root
        self._texts = _LABELS[language]
        self._on_select = on_select
        self._current_id = None
        self._slots = []
        self.no_rule_label = self._texts["none"]
        self.slot_labels = []
        self.title(self._texts["title"])
        self.transient(parent)
        self.resizable(False, False)
        self.configure(fg_color="#181825")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _event: self.destroy())
        content = ctk.CTkFrame(self, fg_color="transparent", width=394)
        content.pack(padx=20, pady=16, fill="both", expand=True)
        self.heading = ctk.CTkLabel(content, text=self._texts["title"], font=ctk.CTkFont(size=18, weight="bold"))
        self.heading.pack(anchor="w")
        self.slot_menu = ctk.CTkOptionMenu(content, values=[self.no_rule_label], command=self._select_slot)
        self.slot_menu.pack(fill="x", pady=(12, 0))
        name_label = ctk.CTkLabel(content, text=self._texts["name"])
        name_label.pack(anchor="w", pady=(10, 2))
        self.name_text = StringVar(self)
        self.name_entry = ctk.CTkEntry(content, textvariable=self.name_text)
        self.name_entry.pack(fill="x")
        content_label = ctk.CTkLabel(content, text=self._texts["content"])
        content_label.pack(anchor="w", pady=(10, 2))
        self.rule_text = ctk.CTkTextbox(content, width=380, height=220)
        self.rule_text.pack(fill="both", expand=True)
        ctk.CTkLabel(content, text=self._texts["hint"], text_color="#bac2de", wraplength=380, justify="left").pack(anchor="w", pady=(8, 2))
        self.status = ctk.CTkLabel(content, text="", text_color="#f38ba8", wraplength=380)
        self.status.pack(fill="x")
        self.save_button = ctk.CTkButton(content, text=self._texts["save"], command=self._save)
        self.save_button.pack(fill="x")
        if appearance is not None:
            for label in (self.heading, name_label, content_label):
                label.configure(text_color=appearance.ink)
            self.slot_menu.configure(
                fg_color=appearance.control, button_color=appearance.hover,
                button_hover_color=appearance.selected, dropdown_hover_color=appearance.control,
            )
            self.name_entry.configure(fg_color="#20202e", border_color=appearance.border)
            self.rule_text.configure(
                fg_color="#20202e", border_width=1, border_color=appearance.border,
                scrollbar_button_color=appearance.control, scrollbar_button_hover_color=appearance.hover,
            )
            self.save_button.configure(fg_color=appearance.control, hover_color=appearance.hover)
        self._reload()
        match = next((i for i, rule in enumerate(self._slots) if selected and rule.id == selected.id), None)
        if match is not None:
            self._display_snapshot(self._slots[match], match)
        else:
            self._display_new()
            if selected is not None:
                self._set_content(selected.name, selected.text)
        self.update_idletasks()
        self.geometry(f"+{parent.winfo_x() + 25}+{parent.winfo_y() + 35}")
        if not restore_parent_grab:
            set_dark_title_bar(self)
        self.grab_set()
        self.focus_set()

    def _reload(self):
        try:
            self._slots = load_rule_slots(self._root_path)
        except (ValueError, OSError) as error:
            self._slots = []
            self.status.configure(text=f"{self._texts['error']}\n{error}", text_color="#f38ba8")
        self.slot_labels = [f"{self._texts['slot']} {i} · {rule.name}" for i, rule in enumerate(self._slots, 1)]
        self.slot_menu.configure(values=[self.no_rule_label, self._texts["new"], *self.slot_labels])

    def _set_content(self, name, text):
        self.name_text.set(name)
        self.rule_text.delete("1.0", "end")
        self.rule_text.insert("1.0", text)

    def _display_new(self):
        self._current_id = None
        self.slot_menu.set(self._texts["new"])
        self.heading.configure(text=self._texts["new"])
        self._set_content("", "")

    def _display_snapshot(self, rule, index=None):
        self._current_id = rule.id
        label = self.slot_labels[index] if index is not None else rule.name
        self.slot_menu.set(label)
        self.heading.configure(text=label)
        self._set_content(rule.name, rule.text)

    def _select_slot(self, label):
        self.status.configure(text="")
        if label == self.no_rule_label:
            self._current_id = None
            self.slot_menu.set(label)
            self.heading.configure(text=label)
            self._set_content("", "")
            self._on_select(None)
        elif label == self._texts["new"]:
            self._display_new()
        else:
            index = self.slot_labels.index(label)
            rule = self._slots[index]
            self._display_snapshot(rule, index)
            self._on_select(rule)

    def _save(self):
        try:
            rule = save_rule_slot(self._root_path, name=self.name_text.get(),
                                  text=self.rule_text.get("1.0", "end-1c"), slot_id=self._current_id)
        except (ValueError, OSError) as error:
            self.status.configure(text=f"{self._texts['error']}\n{error}", text_color="#f38ba8")
            return
        self._reload()
        index = next((i for i, item in enumerate(self._slots) if item.id == rule.id), None)
        self._display_snapshot(rule, index)
        self._on_select(rule)
        self.status.configure(text=self._texts["saved"].format(name=rule.name), text_color="#a6e3a1")

    def destroy(self):
        # CustomTkinter schedules title-bar and focus updates after showing a Toplevel.
        # Cancel this window's timers before destruction when it is closed immediately.
        for timer in self.tk.call("after", "info"):
            command, _ = self.tk.call("after", "info", timer)
            if command in (self._tclCommands or ()):
                self.after_cancel(timer)
        super().destroy()
        if self._restore_parent_grab and self._parent.winfo_exists():
            self._parent.grab_set()
            self._parent.focus_set()
