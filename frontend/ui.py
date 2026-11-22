import os
from functools import partial
from pathlib import Path
import queue
import tkinter as tk
import customtkinter as ctk
from PIL import Image
import threading
import keyboard
from win32api import GetSystemMetrics
from win32gui import error as Win32Error

from core import configure_runtime
from core.config import APP_ICON_PATH, APP_NAME, ASSETS_DIR
from core.live_translation import WindowCrop
from core.ocr_settings import FLOW_DIRECT
from core.setup_models import available_model_settings, model_items
from core.window_capture import list_windows, open_window_capture
from frontend.live_overlay import LiveOverlay, RegionPicker, primary_screen_region, set_capture_exclusion
from frontend.ocr_settings_window import OCRSettingsWindow
from frontend.windows_shell import set_process_app_id, set_taskbar_window

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
_ROOT = Path(__file__).resolve().parents[1]

# Từ điển hỗ trợ đa ngôn ngữ
LANG_TEXTS = {
    "VI": {
        "sub_logo": "Trình dịch Desktop",
        "btn_area": " Chọn vùng",
        "btn_clear": " Xóa đè",
        "btn_settings": " Cài đặt",
        "btn_setup": "Thiết lập",
        "btn_quit": " Đóng ứng dụng",
        "status_title": "TRẠNG THÁI",
        "auto_title": "TỰ ĐỘNG DỊCH",
        "auto_switch": "Theo dõi",
        "area_fullscreen": "Vùng: Toàn màn hình",
        "area_prefix": "Phân vùng quét",
        "history_title": "BẢNG ĐỐI CHIẾU CÂU DỊCH",
        "history_orig": "GỐC",
        "history_trans": "DỊCH",
        "status_ready": "Sẵn sàng",
        "live_capture_desktop": "Dịch vùng màn hình",
        "live_capture_window": "Dịch cửa sổ để quay",
        "btn_pick_window": "Chọn cửa sổ",
        "btn_whole_window": "Toàn cửa sổ",
        "window_crop_set": "Đã chọn vùng",
        "status_focus_window": "Đưa cửa sổ nguồn ra trước rồi chọn lại vùng",
        "status_live_focus_window": "Chuyển sang cửa sổ nguồn để xem bản dịch",
        "status_pick_window": "Hãy chọn cửa sổ để dịch",
        "status_no_windows": "Không tìm thấy cửa sổ để dịch",
        "status_restore_window": "Mở cửa sổ nguồn rồi bấm Bắt đầu",
        "status_window_paused": "Đang chờ cửa sổ nguồn",
        "btn_live_start": " Bắt đầu (`)",
        "btn_live_stop": " Dừng (`)",
        "btn_fullscreen": "Toàn màn hình",
        "status_live_waiting": "Đang nhận dạng văn bản…",
        "status_live_running": "Đang dịch trực tiếp",
        "status_live_error": "Lỗi dịch trực tiếp",
        "status_missing_models": "Mở Thiết lập để tải các mục còn thiếu",
        "pick_area": "Kéo để chọn vùng dịch",
        "btn_cancel": "Hủy",
    },
    "EN": {
        "sub_logo": "Desktop Translator",
        "btn_area": " Select Area",
        "btn_clear": " Clear Overlay",
        "btn_settings": " Settings",
        "btn_setup": "Setup",
        "btn_quit": " Quit Application",
        "status_title": "STATUS",
        "auto_title": "AUTO TRANSLATE",
        "auto_switch": "Monitor",
        "area_fullscreen": "Area: Fullscreen",
        "area_prefix": "Scan area",
        "history_title": "TRANSLATION COMPARISON",
        "history_orig": "ORIGINAL",
        "history_trans": "TRANSLATION",
        "status_ready": "Ready",
        "live_capture_desktop": "Translate screen area",
        "live_capture_window": "Translate window for recording",
        "btn_pick_window": "Choose window",
        "btn_whole_window": "Whole window",
        "window_crop_set": "Area selected",
        "status_focus_window": "Bring the source window forward, then select the area again",
        "status_live_focus_window": "Switch to the source window to see translations",
        "status_pick_window": "Choose a window to translate",
        "status_no_windows": "No windows available to translate",
        "status_restore_window": "Restore the source window, then press Start",
        "status_window_paused": "Waiting for the source window",
        "btn_live_start": " Start (`)",
        "btn_live_stop": " Stop (`)",
        "btn_fullscreen": "Full screen",
        "status_live_waiting": "Looking for text…",
        "status_live_running": "Translating live",
        "status_live_error": "Live translation error",
        "status_missing_models": "Open Setup to install the missing items",
        "pick_area": "Drag to select a translation area",
        "btn_cancel": "Cancel",
    }
}


class FloatingWidget(ctk.CTkToplevel):
    """Thanh công cụ nổi nhỏ gọn ở góc màn hình khi thu nhỏ ứng dụng chính."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(fg_color="#181825")
        
        self.geometry("275x40+10+10")

        # Logic kéo thả (Drag and Drop) widget nổi trên màn hình
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._on_drag)

        # Chấm tròn hiển thị trạng thái hoạt động (Xanh/Cam)
        self.status_dot = ctk.CTkLabel(
            self, text="●", font=ctk.CTkFont(size=14), text_color="#a6e3a1"
        )
        self.status_dot.pack(side="left", padx=(10, 5))

        # Khởi tạo nhãn trạng thái dựa trên ngôn ngữ hiện tại của cha
        initial_status = self.parent.status_label.cget("text").splitlines()[0]
        self.status_text = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#cdd6f4"
        )
        self.status_text.pack(side="left", padx=2)
        self.update_status(initial_status, self.parent.status_label.cget("text_color"))

        # Nút khôi phục lại giao diện lớn
        self.btn_restore = ctk.CTkButton(
            self,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "open_icon.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "open_icon.png")),
                size=(18, 18)
            ),
            text="", width=26, height=26, corner_radius=13,
            fg_color="#313244", hover_color="#45475a",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.parent.restore_dashboard
        )
        self.btn_restore.pack(side="right", padx=(2, 5))

        # Nút chọn vùng quét trực tiếp trên widget nổi
        self.btn_area = ctk.CTkButton(
            self,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "zone.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "zone.png")),
                size=(16, 16)
            ),
            text="",
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#313244",
            hover_color="#45475a",
            command=self.parent.start_area_selection
        )
        self.btn_area.pack(side="right", padx=2)

        # Nút dịch nhanh trực tiếp trên widget nổi
        self.btn_quick_scan = ctk.CTkButton(
            self,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "translate_icon.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "translate_icon.png")),
                size=(18, 18)
            ),
            text="",
            width=26,
            height=26,
            corner_radius=13,
            fg_color="#313244",
            hover_color="#45475a",
            command=self.parent.trigger_scan
        )
        self.btn_quick_scan.pack(side="right", padx=2)

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag(self, event):
        x = self.winfo_x() - self._drag_start_x + event.x
        y = self.winfo_y() - self._drag_start_y + event.y
        self.geometry(f"+{x}+{y}")

    def update_status(self, text, color_hex):
        self.status_text.configure(text=text if len(text) < 27 else f"{text[:24]}…")
        self.status_dot.configure(text_color=color_hex)

    def show_live_activity(self, active):
        self.btn_quick_scan.configure(fg_color="#f38ba8" if active else "#313244")


class ControlPanel(ctk.CTk):
    def __init__(self, *, live_session_factory=None, live_overlay_factory=LiveOverlay):
        configure_runtime()
        set_process_app_id()
        super().__init__()

        self.title(APP_NAME)
        self.geometry("700x520")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color="#11111b")
        self.iconbitmap(str(APP_ICON_PATH))

        self.current_lang = "VI"
        self._desktop_area = None
        self._picker = None
        self._closing = False
        self._ui_actions = queue.SimpleQueue()
        self._status = ("status_ready", "#a6e3a1", "")
        self.floating_widget = None
        self._ocr_settings = available_model_settings(_ROOT)
        self._settings_window = None
        self._live_session_factory = live_session_factory or self._create_live_session
        self._live_overlay_factory = live_overlay_factory
        self._live_session = None
        self._live_overlay = None
        self._live_region = None
        self._live_capture_mode = "desktop"
        self._live_window = None
        self._live_window_crop = None
        self._live_window_paused = False
        self._last_live_pairs = set()
        self._excluded_controls = []

        self._build_ui()
        self._sync_capture_widgets()
        self._update_area_label()
        set_taskbar_window(self)
        self.bind("<Unmap>", self._on_minimize)
        self.bind("<Map>", self._on_restore)
        self.protocol("WM_DELETE_WINDOW", self.quit_all)
        self._translation_hotkey = keyboard.add_hotkey("`", lambda: self.post_ui(self.trigger_scan))
        self._poll_id = self.after(33, self._poll_activity)

    def _create_live_session(self, *, capture_factory=None, frame_crop=None):
        from core.live_models import LiveModelClient
        from core.live_translation import LiveTranslationSession

        fps = os.getenv("MANGALENS_LIVE_CAPTURE_FPS", "30").strip()
        if fps not in {"30", "60"}:
            raise ValueError("MANGALENS_LIVE_CAPTURE_FPS must be 30 or 60.")
        return LiveTranslationSession(
            partial(LiveModelClient, ocr_settings=self._ocr_settings), capture_fps=int(fps),
            capture_factory=capture_factory, frame_crop=frame_crop,
            group_dialogue=self._ocr_settings.flow == FLOW_DIRECT,
        )

    def _create_window_live_session(self, target):
        return self._create_live_session(
            capture_factory=lambda: open_window_capture(target.hwnd), frame_crop=self._live_window_crop,
        )

    def _open_ocr_settings(self):
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.focus_set()
            return
        self._settings_window = OCRSettingsWindow(
            self, _ROOT, self._ocr_settings, self.current_lang, self._apply_ocr_settings,
        )

    def _apply_ocr_settings(self, settings):
        self._stop_live()
        self._close_live_session()
        self._ocr_settings = settings

    def _open_model_setup(self):
        from frontend.setup_window import run_setup

        run_setup(model_items(_ROOT, include_optional=True), parent=self)

    def post_ui(self, action):
        """Queue a UI action without entering Tcl from a worker or keyboard hook."""
        if not self._closing:
            self._ui_actions.put(action)

    def _poll_activity(self):
        self._poll_id = self.after(33, self._poll_activity)
        while not self._closing:
            try:
                action = self._ui_actions.get_nowait()
            except queue.Empty:
                break
            action()
        if not self._closing:
            self._poll_live()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1, minsize=190)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ── 1. SIDEBAR PANEL (Bên trái) ──────────────────────────────────────
        self.sidebar_frame = ctk.CTkFrame(self, fg_color="#181825", corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        with Image.open(ASSETS_DIR / "live-trans-logo.png") as asset:
            logo = ctk.CTkImage(asset.copy(), size=(round(40 * asset.width / asset.height), 40))
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text=APP_NAME, image=logo, compound="left", padx=0,
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#89b4fa"
        )
        self.logo_label.pack(padx=20, pady=(25, 2), anchor="w")
        
        # Nhãn dịch vụ phụ trợ
        self.sub_logo = ctk.CTkLabel(
            self.sidebar_frame, text=LANG_TEXTS[self.current_lang]["sub_logo"], 
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color="#585b70"
        )
        self.sub_logo.pack(padx=20, pady=(0, 10), anchor="w")

        # Nút chuyển đổi ngôn ngữ Việt - Anh
        self.lang_switch = ctk.CTkSegmentedButton(
            self.sidebar_frame,
            values=["VI", "EN"],
            command=self._change_language,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1e1e2e",
            selected_color="#89b4fa",
            unselected_color="#313244",
        )
        self.lang_switch.pack(padx=15, pady=(0, 15), fill="x")
        self.lang_switch.set(self.current_lang)

        self.btn_scan = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "translate_icon_d.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "translate_icon_d.png")),
                size=(20, 20)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_live_start"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color="#a6e3a1",
            text_color="#11111b",
            hover_color="#94e2d5",
            height=40,
            command=self.trigger_scan,
        )
        self.btn_scan.pack(pady=10, padx=15, fill="x")

        self.btn_area = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "zone.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "zone.png")),
                size=(18, 18)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_area"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#313244",
            text_color="#cdd6f4",
            hover_color="#45475a",
            height=32,
            command=self.start_area_selection,
        )
        self.btn_area.pack(pady=5, padx=15, fill="x")
        with Image.open(ASSETS_DIR / "window-transparent.png") as asset:
            window_icon = ctk.CTkImage(asset.copy(), size=(18, 18))
        self.btn_pick_window = ctk.CTkButton(
            self.sidebar_frame, image=window_icon, compound="left", height=32,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#a6adc8", hover_color="#bac2de", text_color="#11111b",
            command=self._pick_live_window,
        )

        self.btn_clear = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "clear.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "clear.png")),
                size=(18, 18)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_clear"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#313244",
            text_color="#cdd6f4",
            hover_color="#45475a",
            height=32,
            command=self.clear_screen,
        )
        self.btn_clear.pack(pady=5, padx=15, fill="x")

        with Image.open(ASSETS_DIR / "settings.png") as asset:
            settings_icon = ctk.CTkImage(asset.copy(), size=(19, 19))
        self.btn_settings = ctk.CTkButton(
            self.sidebar_frame, image=settings_icon, compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_settings"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#313244", text_color="#cdd6f4", hover_color="#45475a", height=32,
            command=self._open_ocr_settings,
        )
        self.btn_settings.pack(pady=5, padx=15, fill="x")
        self.btn_setup = ctk.CTkButton(
            self.sidebar_frame, text=LANG_TEXTS[self.current_lang]["btn_setup"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#313244", text_color="#cdd6f4", hover_color="#45475a", height=32,
            command=self._open_model_setup,
        )
        self.btn_setup.pack(pady=5, padx=15, fill="x")

        self.btn_quit = ctk.CTkButton(
            self.sidebar_frame,
            image=ctk.CTkImage(
                light_image=Image.open(os.path.join(ASSETS_DIR, "close.png")),
                dark_image=Image.open(os.path.join(ASSETS_DIR, "close.png")),
                size=(16, 16)
            ),
            compound="left",
            text=LANG_TEXTS[self.current_lang]["btn_quit"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#f38ba8",
            text_color="#11111b",
            hover_color="#eba0ac",
            height=28,
            command=self.quit_all,
        )
        self.btn_quit.pack(side="bottom", pady=20, padx=15, fill="x")


        # ── 2. MAIN PANEL (Bên phải) ───────────────────────────────────────
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.live_capture_picker = ctk.CTkSegmentedButton(
            self.main_frame, command=self._on_live_capture_selected,
            font=ctk.CTkFont(family="Segoe UI", size=11), height=28,
        )

        self.live_capture_picker.pack(fill="x", pady=(0, 8))

        # Thẻ trạng thái & Tùy chọn nhanh
        self.top_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.top_row.pack(fill="x", pady=(0, 10))

        # Status Box
        self.status_card = ctk.CTkFrame(self.top_row, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.status_card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.card1_title = ctk.CTkLabel(
            self.status_card, text=LANG_TEXTS[self.current_lang]["status_title"], 
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#7f849c"
        )
        self.card1_title.pack(padx=12, pady=(8, 1), anchor="w")
        
        self.status_label = ctk.CTkLabel(
            self.status_card, text=LANG_TEXTS[self.current_lang]["status_ready"],
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color="#a6e3a1", wraplength=235, justify="left",
        )
        self.status_label.pack(padx=12, pady=(0, 8), anchor="w")

        # Auto Switch Box
        self.auto_card = ctk.CTkFrame(self.top_row, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.auto_card.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.card3_title = ctk.CTkLabel(
            self.auto_card, text=LANG_TEXTS[self.current_lang]["auto_title"], 
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#7f849c"
        )
        self.card3_title.pack(padx=12, pady=(8, 3), anchor="w")

        self.auto_switch = ctk.CTkSwitch(
            self.auto_card, text=LANG_TEXTS[self.current_lang]["auto_switch"],
            command=self.handle_auto_switch,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            progress_color="#89b4fa"
        )
        self.auto_switch.pack(padx=12, pady=(0, 8), anchor="w")

        # Thẻ thông tin vùng quét
        self.area_card = ctk.CTkFrame(self.main_frame, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.area_card.pack(fill="x", pady=5)
        
        self.area_info = ctk.CTkLabel(
            self.area_card, text=LANG_TEXTS[self.current_lang]["area_fullscreen"], 
            font=ctk.CTkFont(family="Consolas", size=11), text_color="#a6adc8"
        )
        self.btn_fullscreen = ctk.CTkButton(
            self.area_card, text=LANG_TEXTS[self.current_lang]["btn_fullscreen"],
            command=self._reset_area, width=100, height=25,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#313244", hover_color="#45475a",
        )
        self.btn_fullscreen.pack(side="right", padx=(0, 8), pady=4)
        self.area_info.pack(side="left", padx=12, pady=6)

        # ── KHU VỰC BẢNG ĐỐI CHIẾU CÂU DỊCH (Translation History) ────────────
        self.history_card = ctk.CTkFrame(self.main_frame, fg_color="#1e1e2e", corner_radius=8, border_width=1, border_color="#313244")
        self.history_card.pack(fill="both", expand=True, pady=(10, 5))

        self.history_title = ctk.CTkLabel(
            self.history_card, text=LANG_TEXTS[self.current_lang]["history_title"], 
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#89b4fa"
        )
        self.history_title.pack(padx=15, pady=(10, 5), anchor="w")

        # Bảng Textbox hiển thị log
        self.history_box = ctk.CTkTextbox(
            self.history_card, 
            fg_color="#11111b",
            text_color="#cdd6f4",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            wrap="word",
            corner_radius=6
        )
        self.history_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.history_box.configure(state="disabled")

    def start_area_selection(self):
        if self._picker is not None:
            return
        target = self._live_window if self._live_capture_mode == "window" else None
        if self._live_capture_mode == "window" and target is None:
            self._set_status("status_pick_window", "#f9e2af")
            return
        bounds = None
        if target is not None:
            try:
                bounds = target.bounds()
                if bounds is None:
                    self._set_status("status_restore_window", "#f9e2af")
                    return
                if not target.is_foreground():
                    target.activate()
            except (OSError, RuntimeError, Win32Error) as error:
                if bounds is None:
                    self._set_status("status_live_error", "#f38ba8", str(error))
                else:
                    self._set_status("status_focus_window", "#f9e2af")
                return
        was_live = self._live_region is not None
        dashboard_visible = self.state() == "normal"
        if was_live:
            self._stop_live()
        if dashboard_visible:
            self.withdraw()
        if self.floating_widget:
            self.floating_widget.withdraw()

        def finish(region):
            self._picker = None
            if region is not None:
                if bounds is None:
                    self._set_area(region)
                else:
                    crop = WindowCrop.from_selection(region, bounds)
                    if crop != self._live_window_crop:
                        self._close_live_session()
                        self._live_window_crop = crop
                        self._update_area_label()
            if dashboard_visible:
                self.restore_dashboard()
            elif self.floating_widget:
                self.floating_widget.deiconify()
            if was_live:
                self._start_live()

        texts = LANG_TEXTS[self.current_lang]
        self._picker = RegionPicker(self, finish, texts["pick_area"], texts["btn_cancel"], region=bounds)

    def _pick_live_window(self):
        texts = LANG_TEXTS[self.current_lang]
        try:
            windows = list_windows(os.getpid())
        except (OSError, Win32Error) as error:
            self._set_status("status_live_error", "#f38ba8", str(error))
            return
        if not windows:
            self._set_status("status_no_windows", "#f9e2af")
            return

        self.restore_dashboard()
        picker = ctk.CTkToplevel(self)
        self._picker = picker
        picker.title(texts["btn_pick_window"])
        picker.iconbitmap(str(APP_ICON_PATH))
        picker.geometry("460x340")
        picker.transient(self)
        picker.attributes("-topmost", True)
        rows = ctk.CTkScrollableFrame(picker)
        rows.pack(fill="both", expand=True, padx=12, pady=12)

        def finish(target=None):
            self._picker = None
            picker.destroy()
            if target is not None:
                self._select_live_window(target)

        for target in windows:
            title = target.title if len(target.title) <= 55 else target.title[:54] + "…"
            ctk.CTkButton(rows, text=f"{title}  [{target.pid}]", anchor="w",
                          command=lambda item=target: finish(item)).pack(fill="x", pady=3)
        picker.protocol("WM_DELETE_WINDOW", finish)
        picker.update_idletasks()
        picker.grab_set()
        picker.focus_set()

    def _select_live_window(self, target):
        if self._live_window != target:
            self._close_live_session()
            self._live_window_crop = None
        self._live_window = target
        self._update_area_label()
        self._set_status("status_ready", "#a6e3a1")

    def _on_live_capture_selected(self, label):
        texts = LANG_TEXTS[self.current_lang]
        self._set_live_capture_mode("window" if label == texts["live_capture_window"] else "desktop")

    def _set_live_capture_mode(self, mode):
        if self._live_region is not None or self._picker is not None or mode == self._live_capture_mode:
            return
        self._close_live_session()
        self._live_capture_mode = mode
        self._sync_capture_widgets()
        self._update_area_label()
        self._set_status("status_ready", "#a6e3a1")

    def _change_language(self, lang_code):
        """Hàm thay đổi ngôn ngữ giao diện chính và đồng bộ hóa các nhãn."""
        self.current_lang = lang_code
        texts = LANG_TEXTS[lang_code]

        self.sub_logo.configure(text=texts["sub_logo"])
        self.btn_area.configure(text=texts["btn_area"])
        self.btn_clear.configure(text=texts["btn_clear"])
        self.btn_settings.configure(text=texts["btn_settings"])
        self.btn_setup.configure(text=texts["btn_setup"])
        self.btn_quit.configure(text=texts["btn_quit"])
        self.btn_fullscreen.configure(text=texts["btn_fullscreen"])

        self.card1_title.configure(text=texts["status_title"])
        self.card3_title.configure(text=texts["auto_title"])
        self.auto_switch.configure(text=texts["auto_switch"])
        self.history_title.configure(text=texts["history_title"])

        self._sync_capture_widgets()
        self._update_area_label()
        self._set_status(*self._status)

    def add_history_log(self, original: str, translated: str):
        """Hàm cập nhật câu gốc và câu dịch lên bảng đối chiếu."""
        if not original.strip():
            return
        
        self.history_box.configure(state="normal")
        label_orig = LANG_TEXTS[self.current_lang]["history_orig"]
        label_trans = LANG_TEXTS[self.current_lang]["history_trans"]

        self.history_box.insert("end", f"{label_orig}: {original.strip()}\n")
        self.history_box.insert("end", f"{label_trans}: {translated.strip()}\n")
        self.history_box.insert("end", "──────────────────────\n")
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

    def _on_minimize(self, event):
        if event.widget == self and self.state() == "iconic":
            if self.floating_widget is None:
                self.floating_widget = FloatingWidget(self)
            else:
                self.floating_widget.deiconify()
            if self._live_region is not None and self._live_capture_mode == "desktop":
                try:
                    self._exclude_control(self.floating_widget)
                except RuntimeError as error:
                    self._abort_live(error)
            self._sync_capture_widgets()

    def restore_dashboard(self):
        self.deiconify()

    def _on_restore(self, event):
        if event.widget != self or self.state() != "normal":
            return
        if self.floating_widget:
            self.floating_widget.withdraw()
        if self._live_region is not None and self._live_capture_mode == "desktop":
            try:
                self._exclude_control(self)
            except RuntimeError as error:
                self._abort_live(error)

    def _set_status(self, key, color_hex, detail=""):
        self._status = (key, color_hex, detail)
        display_text = LANG_TEXTS[self.current_lang].get(key, key)
        if detail:
            display_text = f"{display_text}\n{' '.join(detail.split())[:230]}"
        self.status_label.configure(text=display_text, text_color=color_hex)
        if self.floating_widget:
            self.floating_widget.update_status(display_text.splitlines()[0], color_hex)

    @staticmethod
    def fullscreen_region():
        return primary_screen_region()

    def _update_area_label(self):
        area = self._desktop_area
        texts = LANG_TEXTS[self.current_lang]
        if self._live_capture_mode == "window":
            title = self._live_window.title if self._live_window else texts["status_pick_window"]
            label = title if len(title) <= 28 else title[:27] + "…"
            if self._live_window_crop is not None:
                label = f"{label} · {texts['window_crop_set']}"
            self.area_info.configure(text=label)
            self.btn_fullscreen.configure(
                text=texts["btn_whole_window"],
                state="normal" if self._live_window_crop is not None else "disabled",
            )
            return
        label = texts["area_fullscreen"]
        if area is not None:
            x, y, width, height = area
            label = f"{texts['area_prefix']}: {width}×{height} ({x}, {y})"
        self.area_info.configure(text=label)
        self.btn_fullscreen.configure(text=texts["btn_fullscreen"], state="disabled" if area is None else "normal")

    def _reset_area(self):
        if self._live_capture_mode != "window":
            self._set_area(None)
            return
        if self._live_window_crop is None:
            return
        was_live = self._live_region is not None
        if was_live:
            self._stop_live()
        self._close_live_session()
        self._live_window_crop = None
        self._update_area_label()
        if was_live:
            self._start_live()

    def _set_area(self, area):
        if self._desktop_area == area:
            return
        self._desktop_area = area
        self._update_area_label()
        if self._live_region is not None:
            self._stop_live()
            self._start_live()

    def _sync_capture_widgets(self):
        texts = LANG_TEXTS[self.current_lang]
        active = self._live_region is not None
        window_mode = self._live_capture_mode == "window"
        self.live_capture_picker.configure(
            values=[texts["live_capture_desktop"], texts["live_capture_window"]],
            state="disabled" if active else "normal",
        )
        self.live_capture_picker.set(texts[f"live_capture_{self._live_capture_mode}"])
        if window_mode:
            self.btn_pick_window.configure(text=texts["btn_pick_window"], state="disabled" if active else "normal")
            if not self.btn_pick_window.winfo_manager():
                self.btn_pick_window.pack(before=self.btn_area, pady=5, padx=15, fill="x")
        else:
            self.btn_pick_window.pack_forget()
        label = "btn_live_stop" if active else "btn_live_start"
        self.btn_scan.configure(
            text=texts[label], fg_color="#f38ba8" if active else "#a6e3a1",
            hover_color="#eba0ac" if active else "#94e2d5",
        )
        if self.floating_widget:
            self.floating_widget.show_live_activity(active)

    def _set_monitor(self, active):
        if active:
            self.auto_switch.select()
        else:
            self.auto_switch.deselect()

    def trigger_scan(self):
        if self._picker is not None:
            return
        if self._live_region is None:
            self._start_live()
        else:
            self._stop_live()

    def handle_auto_switch(self):
        if self.auto_switch.get():
            self._start_live()
        else:
            self._stop_live()

    def clear_screen(self):
        if self._live_region is not None:
            self._live_session.clear()
            self._live_overlay.clear()

    def _start_live(self):
        if self._live_region is not None or self._picker is not None:
            self._set_monitor(self._live_region is not None)
            return
        if self._ocr_settings.flow == FLOW_DIRECT:
            missing = [item.label for item in model_items(_ROOT, self._ocr_settings) if not item.is_ready()]
            if missing:
                self._set_monitor(False)
                self._set_status("status_missing_models", "#f9e2af", ", ".join(missing))
                return
        window_mode = self._live_capture_mode == "window"
        overlay_visible = True
        if window_mode and self._live_window is None:
            self._set_monitor(False)
            self._set_status("status_pick_window", "#f9e2af")
            return
        try:
            if self._live_overlay is None:
                self._live_overlay = self._live_overlay_factory(self)
            if window_mode:
                region = self._live_window.bounds()
                if region is None:
                    self._set_monitor(False)
                    self._set_status("status_restore_window", "#f9e2af")
                    return
                self._live_overlay.prepare(region, exclude_from_capture=False)
                if not self._live_window.is_foreground():
                    try:
                        self._live_window.activate()
                    except (OSError, Win32Error):
                        # Windows may deny a foreground change. Polling will show the overlay on focus.
                        pass
                overlay_visible = self._live_window.is_foreground()
                self._live_overlay.set_visible(overlay_visible)
                if self._live_session is None:
                    self._live_session = self._create_window_live_session(self._live_window)
            else:
                region = self._desktop_area or self.fullscreen_region()
                self._live_overlay.prepare(region)
                self._exclude_control(self)
                if self.floating_widget:
                    self._exclude_control(self.floating_widget)
                if self._live_session is None:
                    self._live_session = self._live_session_factory()
            self._live_session.start(region)
        except Exception as error:
            self._abort_live(error)
            return

        self._live_region = region
        self._set_monitor(True)
        self._sync_capture_widgets()
        self._set_status("status_live_waiting" if overlay_visible else "status_live_focus_window", "#f9e2af")

    def _stop_live(self):
        if self._live_session is not None and self._live_region is not None:
            self._live_session.stop()
        self._live_region = None
        self._live_window_paused = False
        self._last_live_pairs.clear()
        if self._live_overlay is not None:
            self._live_overlay.hide()
        for control in self._excluded_controls:
            try:
                set_capture_exclusion(control, False)
            except (OSError, RuntimeError, Win32Error):
                pass
        self._excluded_controls.clear()
        self._set_monitor(False)
        self._sync_capture_widgets()
        self._set_status("status_ready", "#a6e3a1")

    def _exclude_control(self, control):
        if control in self._excluded_controls or control.state() != "normal":
            return
        control.update_idletasks()
        try:
            set_capture_exclusion(control)
        except (OSError, RuntimeError, Win32Error) as error:
            raise RuntimeError(f"Could not exclude {APP_NAME} controls from capture. {error}") from error
        self._excluded_controls.append(control)

    def _abort_live(self, error):
        self._stop_live()
        if self._live_capture_mode == "window":
            self._close_live_session()
        self._set_status("status_live_error", "#f38ba8", str(error))

    def _poll_live(self):
        if self._live_region is None:
            return
        overlay_visible = True
        if self._live_capture_mode == "window":
            try:
                overlay_visible = self._track_live_window()
            except (OSError, RuntimeError, Win32Error) as error:
                self._abort_live(error)
                return
            if overlay_visible is None:
                return
            if not overlay_visible and self._status[0] != "status_live_focus_window":
                self._set_status("status_live_focus_window", "#f9e2af")
            elif overlay_visible and self._status[0] == "status_live_focus_window":
                if self._last_live_pairs:
                    self._set_status("status_live_running", "#a6e3a1")
                else:
                    self._set_status("status_live_waiting", "#f9e2af")
        update = self._live_session.poll()
        if update is None or update.region != self._live_region:
            return
        if update.error:
            self._abort_live(update.error)
            return

        self._live_overlay.render(update.region, update.lines)
        if overlay_visible:
            self._set_status("status_live_running", "#a6e3a1")
        pairs = set()
        for line in update.lines:
            pair = (line.source, line.translation)
            if pair not in pairs and pair not in self._last_live_pairs:
                self.add_history_log(*pair)
            pairs.add(pair)
        self._last_live_pairs = pairs

    def _track_live_window(self):
        region = self._live_window.bounds()
        if region is None:
            if not self._live_window_paused:
                self._live_session.stop()
                self._live_overlay.hide()
                self._live_window_paused = True
                self._set_status("status_window_paused", "#f9e2af")
            return None
        if self._live_window_paused or region != self._live_region:
            self._live_overlay.prepare(region, exclude_from_capture=False)
            self._live_session.start(region)
            self._live_region = region
            self._live_window_paused = False
            self._set_status("status_live_waiting", "#f9e2af")
        overlay_visible = self._live_window.is_foreground()
        self._live_overlay.set_visible(overlay_visible)
        return overlay_visible

    def _close_live_session(self):
        if self._live_session is not None:
            self._live_session.close()
            self._live_session = None

    def quit_all(self):
        if self._closing:
            return
        self._closing = True
        self.after_cancel(self._poll_id)
        if self._translation_hotkey is not None:
            keyboard.remove_hotkey(self._translation_hotkey)
        if self._picker is not None:
            if isinstance(self._picker, RegionPicker):
                self._picker.close()
            else:
                self._picker.destroy()
        self._close_live_session()
        if self._live_overlay is not None:
            self._live_overlay.close()
        self.destroy()
