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
