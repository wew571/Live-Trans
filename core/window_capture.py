"""Capture one Windows window independently of the desktop subtitle layers."""

import ctypes
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass
from threading import Event, Lock
from time import monotonic

import numpy as np
import win32con
import win32gui
import win32process


_EXTENDED_FRAME_BOUNDS = 9
_CLOAKED = 14
_FRAME_TIMEOUT_SECONDS = 5


def _dwm_attribute(hwnd, attribute, value):
    get_attribute = ctypes.windll.dwmapi.DwmGetWindowAttribute
    get_attribute.argtypes = (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
    result = get_attribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
    if result < 0:
        raise OSError(f"Could not read window bounds or visibility (0x{result & 0xffffffff:08x}).")
    return value


@dataclass(frozen=True)
class WindowTarget:
    hwnd: int
    pid: int
    title: str

    def bounds(self):
        if not win32gui.IsWindow(self.hwnd):
            raise RuntimeError("The selected window has closed. Choose a window again.")
        if win32process.GetWindowThreadProcessId(self.hwnd)[1] != self.pid:
            raise RuntimeError("The selected window has changed. Choose a window again.")
        if not win32gui.IsWindowVisible(self.hwnd) or win32gui.IsIconic(self.hwnd):
            return None
        if _dwm_attribute(self.hwnd, _CLOAKED, wintypes.DWORD()).value:
            return None
        rect = _dwm_attribute(self.hwnd, _EXTENDED_FRAME_BOUNDS, wintypes.RECT())
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width > 0 and height > 0:
            return rect.left, rect.top, width, height
        return None

    def is_foreground(self):
        return win32gui.GetForegroundWindow() == self.hwnd

    def activate(self):
        win32gui.SetForegroundWindow(self.hwnd)


def list_windows(exclude_pid):
    windows = []

    def collect(hwnd, _):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW:
                return True
            pid = win32process.GetWindowThreadProcessId(hwnd)[1]
            title = win32gui.GetWindowText(hwnd).strip()
            if title and pid != exclude_pid and not _dwm_attribute(hwnd, _CLOAKED, wintypes.DWORD()).value:
                windows.append(WindowTarget(hwnd, pid, title))
        except (OSError, win32gui.error):
            # A window can disappear while EnumWindows is visiting it.
            pass
        return True

    win32gui.EnumWindows(collect, None)
    return sorted(windows, key=lambda window: window.title.casefold())


class _WindowFrames:
    def __init__(self):
        self._lock = Lock()
        self._latest = None
        self._closed = Event()
        self._started_at = monotonic()
        self._received_frame = False

    def on_frame_arrived(self, frame, _control):
        with self._lock:
            self._latest = frame.frame_buffer

    def on_closed(self):
        self._closed.set()

    def grab(self, region):
        if self._closed.is_set():
            raise RuntimeError("Windows stopped capturing the selected window.")
        with self._lock:
            frame, self._latest = self._latest, None
        if frame is not None and frame.shape[:2] == (region[3], region[2]):
            self._received_frame = True
            # The native frame owns a mapped texture. Return independent BGR pixels.
            return np.ascontiguousarray(frame[:, :, :3])
        # Frames from the previous size can arrive during a resize.
        if not self._received_frame and monotonic() - self._started_at > _FRAME_TIMEOUT_SECONDS:
            raise RuntimeError("No matching frames from the selected window. Try windowed or borderless mode.")
        return None


@contextmanager
def open_window_capture(hwnd):
    try:
        from windows_capture import WindowsCapture
    except ImportError as error:
        raise RuntimeError("Window capture requires windows-capture==2.0.1 in the app environment.") from error

    frames = _WindowFrames()
    capture = WindowsCapture(window_hwnd=hwnd, cursor_capture=False)
    capture.event(frames.on_frame_arrived)
    capture.event(frames.on_closed)
    control = capture.start_free_threaded()
    try:
        yield frames
    finally:
        if not control.is_finished():
            control.stop()
