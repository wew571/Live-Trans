"""Run the preflight and installer before loading application dependencies."""

import errno
import logging
import math
import tkinter as tk
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from time import monotonic
from tkinter import ttk

from core.config import APP_ICON_PATH, APP_NAME
from core.setup_types import DownloadInterrupted, InstallationItem
from frontend.windows_shell import set_process_app_id


_BACKOFF_SECONDS = (2, 4, 8, 16, 30)
_BG = "#11111b"
_PANEL = "#181825"
_FG = "#cdd6f4"
_MUTED = "#a6adc8"
_BLUE = "#89b4fa"
_LOG_MAX_LINES = 1000


@dataclass
class _ScanResult:
    missing: tuple[InstallationItem, ...]
    errors: dict[str, Exception]


def _scan(items: Sequence[InstallationItem], report=None) -> _ScanResult:
    missing = []
    errors = {}
    for index, item in enumerate(items):
        if report is not None:
            report(f"Đang kiểm tra {item.label}…", index * 100 // len(items))
        try:
            if not item.is_ready():
                missing.append(item)
        except Exception as exc:
            missing.append(item)
            errors[item.key] = exc
    if report is not None:
        report("Đã kiểm tra xong.", 100)
    return _ScanResult(tuple(missing), errors)


def _configure_styles(root: tk.Tk) -> None:
    root.configure(background=_BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=_BG)
    style.configure("Panel.TFrame", background=_PANEL)
    style.configure("TLabel", background=_BG, foreground=_FG, font=("Segoe UI", 10))
    style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
    style.configure("Status.TLabel", foreground=_MUTED, font=("Segoe UI", 9))
    style.configure("Error.TLabel", foreground="#f38ba8", font=("Segoe UI", 9))
    style.configure("Panel.TLabel", background=_PANEL)
    style.configure("Group.TLabel", background=_PANEL, font=("Segoe UI", 12, "bold"))
    style.configure("TCheckbutton", background=_PANEL, foreground=_FG,
                    indicatorbackground="#313244", padding=(3, 4), font=("Segoe UI", 10))
    style.map("TCheckbutton", background=[("active", "#313244")],
              foreground=[("disabled", _MUTED)],
              indicatorbackground=[("selected", _BLUE)])
    style.configure("TButton", background="#313244", foreground=_FG,
                    padding=(9, 6), bordercolor="#45475a", font=("Segoe UI", 9))
    style.map("TButton", background=[("active", "#45475a")], foreground=[("disabled", _MUTED)])
    style.configure("Install.TButton", background=_BLUE, foreground=_BG)
    style.map("Install.TButton", background=[("disabled", "#313244"), ("active", "#b4befe")],
              foreground=[("disabled", _MUTED)])
    style.configure("Horizontal.TProgressbar", background=_BLUE, troughcolor="#313244",
                    bordercolor="#313244", lightcolor=_BLUE, darkcolor=_BLUE, thickness=9)
    style.configure("Vertical.TScrollbar", background="#45475a", troughcolor=_PANEL,
                    arrowcolor=_FG, bordercolor=_PANEL, lightcolor="#45475a", darkcolor="#45475a")


class _Checklist:
    def __init__(self, parent, column, title, action, install, *, row=0, span=1, height=170):
        self.frame = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        padding = (0, 6) if column == 0 else (6, 0)
        self.frame.grid(row=row, column=column, columnspan=span, sticky="nsew",
                        padx=padding if span == 1 else 0, pady=(10, 0) if row else 0)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        ttk.Label(self.frame, text=title, style="Group.TLabel").grid(row=0, column=0, sticky="w")
        self.select_all = ttk.Button(self.frame, text="Chọn tất cả", command=self._toggle_all)
        self.select_all.grid(row=0, column=1, sticky="e", columnspan=2)

        self.canvas = tk.Canvas(self.frame, background=_PANEL, highlightthickness=0, width=275, height=height)
        self.canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 10))
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=1, column=2, sticky="ns", pady=(8, 10))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.rows = ttk.Frame(self.canvas, style="Panel.TFrame")
        window_id = self.canvas.create_window((0, 0), window=self.rows, anchor="nw")
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(window_id, width=event.width))
        self.rows.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<MouseWheel>", self._scroll)
        self.rows.bind("<MouseWheel>", self._scroll)

        self.install = ttk.Button(self.frame, text=action, style="Install.TButton",
                                  command=lambda: install(self.selected()))
        self.install.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.entries = {}
        self.enabled = False

    def update(self, items):
        selected = {item.key for item in self.selected()}
        for widget in self.rows.winfo_children():
            widget.destroy()
        self.entries.clear()
        for item in items:
            variable = tk.BooleanVar(self.rows, value=item.key in selected)
            box = ttk.Checkbutton(self.rows, text=item.label, variable=variable, command=self._sync_buttons)
            box.pack(fill="x")
            box.bind("<MouseWheel>", self._scroll)
            box.bind("<FocusIn>", lambda event: self._show_focused(event.widget))
            self.entries[item.key] = (item, variable, box)
        if not self.entries:
            ttk.Label(self.rows, text="Đã đầy đủ", style="Panel.TLabel").pack(anchor="w", pady=5)
        self._sync_buttons()

    def selected(self):
        return [item for item, variable, _ in self.entries.values() if variable.get()]

    def set_enabled(self, enabled):
        self.enabled = enabled
        for _, _, box in self.entries.values():
            box.state(["!disabled" if enabled else "disabled"])
        self._sync_buttons()

    def _sync_buttons(self):
        selected = self.selected()
        all_selected = bool(self.entries) and len(selected) == len(self.entries)
        self.select_all.configure(text="Bỏ chọn tất cả" if all_selected else "Chọn tất cả")
        self.select_all.state(["!disabled" if self.enabled and self.entries else "disabled"])
        self.install.state(["!disabled" if self.enabled and selected else "disabled"])

    def _toggle_all(self):
        value = len(self.selected()) != len(self.entries)
        for _, variable, _ in self.entries.values():
            variable.set(value)
        self._sync_buttons()

    def _scroll(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _show_focused(self, box):
        visible_top = self.canvas.canvasy(0)
        visible_height = self.canvas.winfo_height()
        top = box.winfo_y()
        bottom = top + box.winfo_height()
        content_height = self.rows.winfo_height()
        if top < visible_top:
            self.canvas.yview_moveto(top / content_height)
        elif bottom > visible_top + visible_height:
            self.canvas.yview_moveto((bottom - visible_height) / content_height)


class _SetupWindow:
    def __init__(self, root, items: Sequence[InstallationItem], executor: ThreadPoolExecutor,
                 finish, *, install_now=False, offer_optional=False):
        self.root = root
        self.items = tuple(items)
        self.executor = executor
        self.finish = finish
        self.install_now = install_now
        self.can_continue = False
        self.offer_optional = offer_optional
        self.messages = SimpleQueue()
        self.log_lines = SimpleQueue()
        self.job: Future | None = None
        self.on_done: Callable | None = None
        self.pending: list[InstallationItem] = []
        self.ready = False
        self.closing = False
        self.retry_index = 0
        self.retry_time: float | None = None
        self.percent: int | None = 0
        self.animating = False

        title = f"Thiết lập {APP_NAME}"
        root.title(title)
        root.minsize(670, 430)
        _configure_styles(root)
        if APP_ICON_PATH.is_file():
            try:
                root.iconbitmap(str(APP_ICON_PATH))
            except tk.TclError:
                pass
        root.protocol("WM_DELETE_WINDOW", self._close)

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=title, style="Title.TLabel").pack(anchor="w")

        progress_row = ttk.Frame(frame)
        progress_row.pack(fill="x", pady=(12, 0))
        self.progress = ttk.Progressbar(progress_row, maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress_text = ttk.Label(progress_row, text="0%", width=6, anchor="e", style="Status.TLabel")
        self.progress_text.pack(side="right")
        self.status = ttk.Label(frame, text="Đang kiểm tra…", style="Status.TLabel", wraplength=610)
        self.status.pack(anchor="w", pady=(5, 0))

        self.groups = ttk.Frame(frame)
        self.groups.columnconfigure((0, 1), weight=1, uniform="setup-groups")
        self.groups.rowconfigure(0, weight=1)
        self.libraries = _Checklist(self.groups, 0, "Thư viện Python", "Cài thư viện đã chọn", self._install)
        self.models = _Checklist(self.groups, 1, "Mô hình", "Cài mô hình đã chọn", self._install)
        if install_now:
            self.libraries.frame.grid_remove()
            self.models.frame.grid_configure(column=0, columnspan=2, padx=0)
        self.optional = _Checklist(
            self.groups, 0, "Mô hình bổ sung", "Cài mô hình bổ sung đã chọn", self._install,
            row=1, span=2, height=105,
        )
        self.optional.frame.grid_remove()

        self.log_panel = ttk.Frame(frame, style="Panel.TFrame", padding=10)
        self.log_panel.columnconfigure(0, weight=1)
        ttk.Label(self.log_panel, text="Nhật ký cài thư viện", style="Group.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6),
        )
        self.terminal = tk.Text(
            self.log_panel, width=1, height=8, wrap="char", state="disabled", takefocus=True,
            background=_BG, foreground=_FG, selectbackground="#45475a", selectforeground=_FG,
            font=("Consolas", 9), relief="flat", borderwidth=0, highlightthickness=0, padx=8, pady=6,
        )
        self.terminal.grid(row=1, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(self.log_panel, orient="vertical", command=self.terminal.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.terminal.configure(yscrollcommand=scrollbar.set)

        self.details = ttk.Label(frame, style="Error.TLabel", wraplength=610)
        self.details.pack(anchor="w", pady=(10, 0))
        self.retry = ttk.Button(frame, text="Thử lại")
        self.continue_button = ttk.Button(frame, text="Tiếp tục", command=self._complete, style="Install.TButton")

        self.scan_timer = root.after_idle(self._start_scan)
        self.poll_timer = root.after(25, self._poll)

    def stop_timers(self):
        self.root.after_cancel(self.scan_timer)
        self.root.after_cancel(self.poll_timer)
        self.progress.stop()

    def _start_scan(self):
        if self.closing:
            return
        self.retry.pack_forget()
        self.details.configure(text="")
        self._set_status("Đang kiểm tra…", 0)
        self._set_enabled(False)
        self._submit(lambda: _scan(self.items, self._report), self._finish_scan)

    def _finish_scan(self, result):
        required = [item for item in result.missing if not item.optional]
        self.offer_optional = self.offer_optional or bool(required)
        if not result.missing or not self.offer_optional:
            self._complete()
            return
        self._display_missing(result)
        if result.errors:
            item = next(item for item in result.missing if item.key in result.errors)
            self._show_failure(item, result.errors[item.key], checking=True)
        elif self.install_now:
            self.install_now = False
            self._install(required)
        else:
            status = ("Chọn các mục còn thiếu để cài đặt." if required
                      else "Đã đủ mục bắt buộc. Bạn có thể cài thêm mô hình hoặc tiếp tục.")
            self._set_status(status, 100)
            self._set_enabled(True)

    def _install(self, selected):
        if not selected or self.job is not None or self.retry_time is not None or self.closing:
            return
        self.pending = list(selected)
        self.retry_index = 0
        self._start_attempt(resuming=False)

    def _start_attempt(self, *, resuming):
        self.retry_time = None
        self.retry.pack_forget()
        self.details.configure(text="")
        self._set_enabled(False)
        item = self.pending[0]
        action = "Đang cài thư viện" if item.category == "library" else "Đang tải mô hình"
        activity = f"{action} {item.label}…"
        initial_percent = self.percent if resuming else None
        self._set_status(activity, initial_percent)
        if item.category == "library":
            if not self.log_panel.winfo_manager():
                self.log_panel.pack(before=self.details, fill="x", pady=(12, 0))
                self.root.minsize(670, 560)
            self._append_log([f"\n{activity}"])

        def attempt():
            last_percent = initial_percent

            def progress(completed, total):
                nonlocal last_percent
                if item.category != "model":
                    return
                percent = max(0, min(100, completed * 100 // total)) if total and total > 0 else None
                if percent != last_percent:
                    last_percent = percent
                    self._report(activity, percent)

            error = None
            try:
                if item.install_with_log is None:
                    item.install(progress)
                else:
                    item.install_with_log(progress, self.log_lines.put)
            except Exception as exc:
                error = exc
            return _scan(self.items), error

        self._submit(attempt, self._finish_attempt)

    def _finish_attempt(self, outcome):
        result, error = outcome
        item = self.pending[0]
        missing_keys = {entry.key for entry in result.missing}
        self.pending = [entry for entry in self.pending if entry.key in missing_keys]
        self._display_missing(result)
        if item.key in missing_keys and isinstance(error, DownloadInterrupted):
            if item.category == "library":
                self._append_log(["Kết nối bị gián đoạn. Đang chờ thử lại…"])
            self._pause_for_retry(item)
            return
        if error is None and item.key in missing_keys:
            error = result.errors.get(item.key) or RuntimeError("Mục này vẫn còn thiếu sau khi cài đặt.")
        if item.category == "library":
            action = "Không thể cài đặt" if error is not None else "Đã cài đặt"
            self._append_log([f"{action} {item.label}."])
        if error is not None:
            self._show_failure(item, error)
        elif self.pending:
            self.retry_index = 0
            self._start_attempt(resuming=False)
        else:
            self._start_scan()

    def _pause_for_retry(self, item):
        delay = _BACKOFF_SECONDS[min(self.retry_index, len(_BACKOFF_SECONDS) - 1)]
        self.retry_index += 1
        self.retry_time = monotonic() + delay
        self._set_progress(self.percent, animate=False)
        self._update_countdown(item, delay)
        self._show_retry("Thử ngay", lambda: self._start_attempt(resuming=True))

    def _update_countdown(self, item, seconds):
        self.status.configure(text=f"Kết nối bị gián đoạn khi tải {item.label}. Thử lại sau {seconds} giây…")

    def _show_failure(self, item, error, *, checking=False):
        action = "kiểm tra" if checking else "cài đặt"
        self._set_status(f"Không thể {action} {item.label}.", self.percent, animate=False)
        if isinstance(error, PermissionError):
            hint = "Kiểm tra quyền truy cập thư mục cài đặt."
        elif isinstance(error, OSError) and error.errno == errno.ENOSPC:
            hint = "Giải phóng dung lượng ổ đĩa rồi thử lại."
        elif checking:
            hint = "Kiểm tra đường dẫn và quyền truy cập rồi kiểm tra lại."
        elif item.category == "library":
            hint = "Kiểm tra môi trường Python, kết nối mạng và quyền ghi rồi thử lại."
        else:
            hint = "Kiểm tra kết nối mạng và dung lượng ổ đĩa rồi thử lại."
        detail = str(error).strip() or type(error).__name__
        if len(detail) > 360:
            detail = "…" + detail[-360:]
        self.details.configure(text=f"{hint}\n{detail}")
        self._set_enabled(True)
        if not checking and self.pending:
            self._show_retry("Thử lại", lambda: self._install(self.pending))
        else:
            self._show_retry("Kiểm tra lại", self._start_scan)

    def _display_missing(self, result):
        self.can_continue = all(item.optional for item in result.missing)
        self.libraries.update(item for item in result.missing if item.category == "library")
        self.models.update(item for item in result.missing if item.category == "model" and not item.optional)
        self.optional.update(item for item in result.missing if item.category == "model" and item.optional)
        if self.optional.entries:
            self.optional.frame.grid()
            if not self.log_panel.winfo_manager():
                self.root.minsize(670, 570)
        else:
            self.optional.frame.grid_remove()
        if self.can_continue and self.offer_optional:
            self.continue_button.pack(anchor="e", pady=(10, 0))
        else:
            self.continue_button.pack_forget()
        if result.missing and not self.groups.winfo_manager():
            self.groups.pack(before=self.details, fill="both", expand=True, pady=(16, 0))

    def _show_retry(self, label, command):
        self.retry.configure(text=label, command=command)
        self.retry.pack(anchor="w", pady=(8, 0))

    def _set_enabled(self, enabled):
        self.libraries.set_enabled(enabled)
        self.models.set_enabled(enabled)
        self.optional.set_enabled(enabled)
        self.continue_button.state(["!disabled" if enabled and self.can_continue else "disabled"])

    def _set_status(self, label, percent, *, animate=True):
        self.status.configure(text=label)
        self._set_progress(percent, animate=animate)

    def _set_progress(self, percent, *, animate=True):
        self.percent = percent
        if percent is None:
            self.progress_text.configure(text="")
            if animate and not self.animating:
                self.progress.configure(mode="indeterminate", value=0)
                self.progress.start(16)
                self.animating = True
            elif not animate and self.animating:
                self.progress.stop()
                self.animating = False
        else:
            if self.animating:
                self.progress.stop()
                self.animating = False
            self.progress.configure(mode="determinate", value=percent)
            self.progress_text.configure(text=f"{percent}%")

    def _submit(self, work, finished):
        self.on_done = finished
        self.job = self.executor.submit(work)

    def _report(self, label, percent):
        self.messages.put((label, percent))

    def _append_log(self, lines):
        follow = self.terminal.yview()[1] >= 1.0
        self.terminal.configure(state="normal")
        self.terminal.insert("end", "\n".join(lines) + "\n")
        excess = int(self.terminal.index("end-1c").split(".")[0]) - 1 - _LOG_MAX_LINES
        if excess > 0:
            self.terminal.delete("1.0", f"{excess + 1}.0")
        self.terminal.configure(state="disabled")
        if follow:
            self.terminal.see("end-1c")

    def _poll(self):
        job_finished = self.job is not None and self.job.done()
        while True:
            try:
                label, percent = self.messages.get_nowait()
            except Empty:
                break
            if not self.closing:
                self._set_status(label, percent)

        lines = []
        while True:
            try:
                lines.append(self.log_lines.get_nowait())
            except Empty:
                break
        if lines:
            self._append_log(lines)

        if job_finished:
            if self.closing:
                self.finish()
                return
            job, finished = self.job, self.on_done
            self.job = None
            self.on_done = None
            finished(job.result())
            if self.ready:
                return
        elif self.retry_time is not None:
            remaining = math.ceil(self.retry_time - monotonic())
            if remaining <= 0:
                self._start_attempt(resuming=True)
            else:
                self._update_countdown(self.pending[0], remaining)
        self.poll_timer = self.root.after(25, self._poll)

    def _complete(self):
        self._set_status("Đã kiểm tra xong.", 100)
        self.root.update_idletasks()
        self.ready = True
        self.finish()

    def _close(self):
        self.closing = True
        self.retry_time = None
        self.retry.pack_forget()
        self._set_enabled(False)
        if self.job is None or self.job.done():
            self.finish()
        else:
            self._set_status("Đang chờ thao tác hiện tại hoàn tất để đóng…", self.percent, animate=False)


def run_setup(items: Sequence[InstallationItem], *, parent=None, install_now=False) -> bool:
    """Run startup setup or open a modal to review or install missing models."""
    if parent is None:
        set_process_app_id()
        root = tk.Tk()
        finish = root.quit
        wait = root.mainloop
    else:
        previous_grab = parent.grab_current()
        root = tk.Toplevel(parent)
        root.transient(parent)
        closed = tk.BooleanVar(master=root)
        finish = lambda: closed.set(True)
        wait = lambda: root.wait_variable(closed)
        root.bind("<Destroy>", lambda event: finish() if event.widget is root else None)
    try:
        if parent is None:
            def fail_closed(exc_type, exc, traceback):
                logging.getLogger(__name__).error("Startup setup failed", exc_info=(exc_type, exc, traceback))
                finish()
            root.report_callback_exception = fail_closed
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-trans-setup") as executor:
            window = _SetupWindow(
                root, items, executor, finish, install_now=install_now,
                offer_optional=parent is not None and not install_now,
            )
            try:
                if parent is not None:
                    root.grab_set()
                wait()
            finally:
                if root.winfo_exists():
                    window.stop_timers()
            return window.ready and not window.closing
    finally:
        if root.winfo_exists():
            root.destroy()
        if parent is not None and previous_grab is not None and previous_grab.winfo_exists():
            previous_grab.grab_set()
