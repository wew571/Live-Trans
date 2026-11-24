import ctypes
import sys
import tkinter as tk

import win32con
import win32gui
from win32api import GetSystemMetrics

from core.ocr_settings import DEFAULT_TRANSLATION_COLOR


_TRANSPARENT_COLOR = "#ff00ff"
_SUBTITLE_BACKGROUND_OPACITY = 0.70
_WDA_EXCLUDEFROMCAPTURE = 0x11


def primary_screen_region():
    """Return the primary display's bounds in the DPI-aware UI's screen pixels."""
    return 0, 0, GetSystemMetrics(0), GetSystemMetrics(1)


def canvas_box(box, region, canvas_size):
    """Project a capture-relative box onto the canvas covering its capture region."""
    _, _, region_width, region_height = region
    canvas_width, canvas_height = canvas_size
    left, top, right, bottom = box
    scale_x = canvas_width / region_width
    scale_y = canvas_height / region_height
    return left * scale_x, top * scale_y, right * scale_x, bottom * scale_y


def set_capture_exclusion(window, enabled=True):
    if enabled and sys.getwindowsversion().build < 19041:
        raise RuntimeError("Capture exclusion requires Windows 10 version 2004 or newer.")

    hwnd = win32gui.GetAncestor(window.winfo_id(), win32con.GA_ROOT)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    set_affinity = user32.SetWindowDisplayAffinity
    set_affinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    get_affinity = user32.GetWindowDisplayAffinity
    get_affinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    affinity = ctypes.c_uint()
    requested = _WDA_EXCLUDEFROMCAPTURE if enabled else 0
    if not set_affinity(hwnd, requested):
        raise ctypes.WinError(ctypes.get_last_error())
    if not get_affinity(hwnd, ctypes.byref(affinity)):
        raise ctypes.WinError(ctypes.get_last_error())
    if affinity.value != requested:
        raise RuntimeError("Windows did not apply the requested capture exclusion.")


class LiveOverlay:
    """Display-only layers that keep subtitle text opaque above translucent backgrounds."""

    def __init__(self, parent):
        self.background_window, self.background_canvas = self._create_layer(parent)
        self.background_window.attributes("-alpha", _SUBTITLE_BACKGROUND_OPACITY)
        self.window, self.canvas = self._create_layer(parent)
        self.region = None
        self._translation_color = DEFAULT_TRANSLATION_COLOR

    def set_translation_color(self, color):
        color = (color or DEFAULT_TRANSLATION_COLOR).lower()
        if color == self._translation_color:
            return
        self._translation_color = color
        # Keep the chosen text color distinct from the foreground layer's transparency key.
        transparent = "#00ff00" if color == _TRANSPARENT_COLOR else _TRANSPARENT_COLOR
        self.window.attributes("-transparentcolor", transparent)
        self.window.configure(background=transparent)
        self.canvas.configure(background=transparent)
        self.canvas.itemconfigure("all", fill=color)

    @staticmethod
    def _create_layer(parent):
        window = tk.Toplevel(parent)
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.attributes("-transparentcolor", _TRANSPARENT_COLOR)
        window.configure(background=_TRANSPARENT_COLOR)
        canvas = tk.Canvas(window, background=_TRANSPARENT_COLOR, bd=0, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        return window, canvas

    def prepare(self, region, *, exclude_from_capture=True):
        self.hide()
        x, y, width, height = region
        try:
            for window in (self.background_window, self.window):
                window.geometry(f"{width}x{height}")
                window.deiconify()
                window.update_idletasks()
                # Tk's client is inside the native top-level wrapper.
                hwnd = win32gui.GetAncestor(window.winfo_id(), win32con.GA_ROOT)
                self._position(hwnd, x, y)
                styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(
                    hwnd, win32con.GWL_EXSTYLE,
                    styles | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE,
                )
                set_capture_exclusion(window, exclude_from_capture)
            self.window.lift(self.background_window)
        except (OSError, RuntimeError, win32gui.error):
            self.hide()
            raise
        self.region = region

    @staticmethod
    def _position(hwnd, x, y):
        # Tk interprets negative geometry offsets relative to the far edge.
        win32gui.SetWindowPos(hwnd, 0, x, y, 0, 0,
                              win32con.SWP_NOSIZE | win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)

    def move(self, region):
        x, y, _, _ = region
        for window in (self.background_window, self.window):
            hwnd = win32gui.GetAncestor(window.winfo_id(), win32con.GA_ROOT)
            self._position(hwnd, x, y)
        self.region = region

    def set_visible(self, visible):
        if visible and self.window.state() == "withdrawn":
            self.background_window.deiconify()
            self.window.deiconify()
            self.window.lift(self.background_window)
        elif not visible and self.window.state() != "withdrawn":
            self.window.withdraw()
            self.background_window.withdraw()

    def render(self, region, lines):
        if region != self.region:
            return
        self.clear()
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        # Tk reports 1px for a canvas before its first layout pass.
        size = (width if width > 1 else region[2], height if height > 1 else region[3])
        for line in lines:
            if not line.translation.strip():
                continue
            left, top, right, bottom = canvas_box(line.box, region, size)
            self._draw_label(line.translation, (left, top, right, bottom), size)

    def _draw_label(self, translation, box, size):
        left, top, right, bottom = box
        canvas_width, canvas_height = size
        font_size = max(11, min(24, round((bottom - top) * 0.75)))
        padding = 3
        width = max(1, min(canvas_width - 2 * padding, max(120, right - left)))
        x = max(padding, min(left + padding, canvas_width - width - padding))
        y = max(padding, top + padding)
        text = self.canvas.create_text(
            x, y, text=translation, anchor="nw", width=width,
            font=("Segoe UI", -font_size), fill=self._translation_color,
        )
        text_x1, text_y1, text_x2, text_y2 = self.canvas.bbox(text)
        offset = max(0, text_y2 + padding - canvas_height)
        self.canvas.move(text, 0, -offset)
        self.background_canvas.create_rectangle(
            max(0, min(left, text_x1 - padding)),
            max(0, min(top, text_y1 - padding - offset)),
            min(canvas_width, max(right, text_x2 + padding)),
            min(canvas_height, max(bottom, text_y2 + padding - offset)),
            fill="#161925", outline="",
        )

    def clear(self):
        self.canvas.delete("all")
        self.background_canvas.delete("all")

    def hide(self):
        self.clear()
        self.window.withdraw()
        self.background_window.withdraw()
        self.region = None

    def close(self):
        self.window.destroy()
        self.background_window.destroy()


class RegionPicker:
    """Asynchronous drag selection within screen bounds."""

    def __init__(self, parent, callback, prompt, cancel_label, *, region=None, preview=None,
                 exclude_from_capture=False):
        self.callback = callback
        self.region = region if region is not None else primary_screen_region()
        self.start = None
        self.rect = None
        self.window = tk.Toplevel(parent)
        if exclude_from_capture:
            self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.5)
        self.window.attributes("-transparentcolor", _TRANSPARENT_COLOR)
        x, y, width, height = self.region
        self.window.geometry(f"{width}x{height}")
        self.canvas = tk.Canvas(
            self.window, bg="#11111b", cursor="crosshair", bd=0, highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            20, 20, anchor="nw", text=prompt, font=("Segoe UI", 14, "bold"), fill="white"
        )
        if preview is None:
            self.canvas.bind("<Button-1>", self._press)
            self.canvas.bind("<B1-Motion>", self._drag)
            self.canvas.bind("<ButtonRelease-1>", self._release)
        else:
            self.canvas.bind("<Configure>", lambda event: self._draw_preview(event, preview))
        cancel = tk.Button(
            self.window, name="cancel", text=cancel_label, command=lambda: self._finish(None),
            bg="#313244", fg="#ffffff", activebackground="#45475a", activeforeground="#ffffff",
            relief="flat", padx=10, pady=3, cursor="arrow",
        )
        cancel.place(relx=1, rely=1, x=-12, y=-12, anchor="se")
        self.window.protocol("WM_DELETE_WINDOW", lambda: self._finish(None))
        if preview is not None:
            self.window.bind("<Escape>", lambda _event: self._finish(None))
        self.window.update_idletasks()
        hwnd = win32gui.GetAncestor(self.window.winfo_id(), win32con.GA_ROOT)
        win32gui.SetWindowPos(hwnd, 0, x, y, 0, 0,
                              win32con.SWP_NOSIZE | win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)
        if exclude_from_capture:
            try:
                set_capture_exclusion(self.window)
            except (OSError, RuntimeError):
                self.close()
                raise
            self.window.deiconify()
        self.window.grab_set()
        self.window.focus_force()

    def _draw_preview(self, event, preview):
        self.canvas.delete("crop-preview")
        for index, crop in preview:
            left, top, right, bottom = crop.box(event.width, event.height)
            self.canvas.create_rectangle(
                left, top, right, bottom, outline="#89b4fa", width=4,
                fill=_TRANSPARENT_COLOR, tags="crop-preview",
            )
            self.canvas.create_text(
                left + 8, top + 5, text=f"Crop {index}", anchor="nw",
                fill="#89b4fa", font=("Segoe UI", 16, "bold"), tags="crop-preview",
            )

    def _press(self, event):
        self.start = (event.x, event.y)
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#89b4fa", width=2, fill=_TRANSPARENT_COLOR,
        )

    def _drag(self, event):
        if self.start is not None:
            self.canvas.coords(self.rect, *self.start, event.x, event.y)

    def _release(self, event):
        if self.start is None:
            return
        x, y, width, height = self.region
        scale_x = width / self.canvas.winfo_width()
        scale_y = height / self.canvas.winfo_height()
        left, right = sorted(round(min(width, max(0, v * scale_x))) for v in (self.start[0], event.x))
        top, bottom = sorted(round(min(height, max(0, v * scale_y))) for v in (self.start[1], event.y))
        result = (x + left, y + top, right - left, bottom - top)
        self._finish(result if right > left and bottom > top else None)

    def _finish(self, region):
        self.close()
        self.callback(region)

    def close(self):
        self.window.destroy()
