"""Register the window and its icon with the Windows taskbar."""

import ctypes
import logging
from pathlib import Path
import subprocess
import sys

from core import ROOT_DIR
from core.config import APP_ICON_PATH, APP_NAME

_APP_ID = "LiveTrans.Desktop"
_LOG = logging.getLogger(__name__)


def set_process_app_id():
    if sys.platform == "win32":
        register = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        register.argtypes = (ctypes.c_wchar_p,)
        register.restype = ctypes.c_long
        result = register(_APP_ID)
        if result < 0:
            _LOG.warning("Could not register the Windows app ID (0x%08X)", result & 0xFFFFFFFF)


def set_taskbar_window(window):
    import pywintypes
    import win32con
    import win32gui
    from win32com.propsys import propsys, pscon

    window.update_idletasks()
    hwnd = win32gui.GetAncestor(window.winfo_id(), win32con.GA_ROOT)
    launcher = subprocess.list2cmdline([sys.executable, str(Path(ROOT_DIR) / "main.py")])
    try:
        store = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
        for property_key, value in (
            (pscon.PKEY_AppUserModel_RelaunchCommand, launcher),
            (pscon.PKEY_AppUserModel_RelaunchDisplayNameResource, APP_NAME),
            (pscon.PKEY_AppUserModel_RelaunchIconResource, f"{APP_ICON_PATH},0"),
            # Set the grouping ID after the relaunch properties are available.
            (pscon.PKEY_AppUserModel_ID, _APP_ID),
        ):
            store.SetValue(property_key, propsys.PROPVARIANTType(value))
    except pywintypes.com_error:
        _LOG.warning("Could not set the Windows taskbar icon", exc_info=True)


def place_window_beside(window, anchor_hwnd):
    """Lay out two Windows dialogs in the work area of the anchor's monitor."""
    import win32api
    import win32con
    import win32gui

    hwnd = win32gui.GetAncestor(window.winfo_id(), win32con.GA_ROOT)
    left, top, right, _ = win32gui.GetWindowRect(anchor_hwnd)
    dialog_left, dialog_top, dialog_right, dialog_bottom = win32gui.GetWindowRect(hwnd)
    monitor = win32api.MonitorFromWindow(anchor_hwnd, win32con.MONITOR_DEFAULTTONEAREST)
    work_left, work_top, work_right, work_bottom = win32api.GetMonitorInfo(monitor)["Work"]
    width, height = dialog_right - dialog_left, dialog_bottom - dialog_top
    anchor_width = right - left
    flags = win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
    anchor_x = left

    if right + width <= work_right:
        x = right
    elif left - width >= work_left:
        x = left - width
    else:
        x = max(work_left, work_right - width)
        if anchor_width + width <= work_right - work_left:
            anchor_x = x - anchor_width

    x = max(work_left, min(x, work_right - width))
    y = max(work_top, min(top, work_bottom - height))
    win32gui.SetWindowPos(anchor_hwnd, win32con.HWND_TOP, anchor_x, top, 0, 0, flags)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, 0, 0, flags)


def raise_settings_window(hwnd):
    import win32con
    import win32gui

    flags = win32con.SWP_NOSIZE | win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)


def set_dark_title_bar(window):
    """Set the Windows caption theme without hiding or reactivating the dialog."""
    from ctypes import wintypes
    import win32con
    import win32gui

    hwnd = wintypes.HWND(win32gui.GetAncestor(window.winfo_id(), win32con.GA_ROOT))
    dark = wintypes.BOOL(1)
    apply = ctypes.windll.dwmapi.DwmSetWindowAttribute
    if apply(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)) != 0:
        apply(hwnd, 19, ctypes.byref(dark), ctypes.sizeof(dark))
