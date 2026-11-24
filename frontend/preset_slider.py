"""A discrete slider for the Live Trans preset selector."""

from math import cos, radians, sin
from random import Random
from time import monotonic
from tkinter import Canvas

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFilter, ImageTk


_DRAG_THRESHOLDS = {3: 2.7, 4: 3.7, 5: 4.9}
_GLOW_COLORS = ("#7979bf", "#5ea978", "#c47c4c")


class PresetSlider(ctk.CTkFrame):
    _ANIMATION_SECONDS = 0.12
    _FRAME_MS = 15

    def __init__(self, parent, *, stop_count, initial_index, command, animation_enabled=False):
        super().__init__(parent, width=359, height=34, fg_color="transparent", corner_radius=0)
        self.pack_propagate(False)
        self._last_index = stop_count - 1
        self._index = initial_index
        self._position = float(initial_index)
        self._animation_id = None
        self._command = command
        self._animation_enabled = animation_enabled
        self._animation_started = monotonic()
        self._animation_origin = self._position
        self._animation_target = self._position
        self._drag_gate = None
        self._arrival_started = 0.0
        self._texture_cache = {}
        self._texture_photo = None
        self._texture_photo_key = None

        self.canvas = Canvas(
            self, background="#242438", borderwidth=0, highlightthickness=1,
            highlightbackground="#242438", highlightcolor="#a6c8ff", takefocus=True,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._draw_slider)
        self.canvas.bind("<Map>", self._on_map)
        self.canvas.bind("<Unmap>", self._on_unmap)
        self.canvas.bind("<Button-1>", self._mouse_press)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._release_drag)
        self.canvas.bind("<Left>", lambda _event: self._select(self._index - 1))
        self.canvas.bind("<Right>", lambda _event: self._select(self._index + 1))
        self.canvas.bind("<Home>", lambda _event: self._select(0))
        self.canvas.bind("<End>", lambda _event: self._select(self._last_index))

    def get(self):
        return self._index

    def set(self, index):
        index = min(self._last_index, max(0, round(index)))
        if index == self._index:
            return
        previous = self._index
        self._index = index
        self._drag_gate = None
        if not self._animation_enabled:
            self._animation_origin = self._animation_target = self._position = float(index)
            self._draw_slider()
            return
        if index >= 3 and index > previous:
            self._arrival_started = monotonic()
        self._animate_to(index)

    def set_surface_color(self, color: str):
        """Match the canvas surround to its containing preset card."""
        self.canvas.configure(background=color, highlightbackground=color)

    def set_animation_enabled(self, enabled):
        if self._animation_enabled == enabled:
            return
        self._cancel_animation()
        self._animation_enabled = enabled
        self._drag_gate = None
        self._arrival_started = 0.0
        self._animation_origin = self._animation_target = self._position = float(self._index)
        self._tick()

    def destroy(self):
        self._cancel_animation()
        super().destroy()

    def _on_map(self, _event=None):
        if self._animation_id is None:
            self._tick()

    def _on_unmap(self, _event=None):
        self._cancel_animation()
        self._drag_gate = None
        self._animation_target = self._index

    def _cancel_animation(self):
        if self._animation_id is not None:
            self.after_cancel(self._animation_id)
            self._animation_id = None

    def _animate_to(self, target):
        self._cancel_animation()
        self._animation_started = monotonic()
        self._animation_origin = self._position
        self._animation_target = target
        self._tick()

    def _tick(self):
        self._animation_id = None
        now = monotonic()
        elapsed = min((now - self._animation_started) / self._ANIMATION_SECONDS, 1.0)
        eased = 1 - (1 - elapsed) ** 3
        self._position = self._animation_origin + (self._animation_target - self._animation_origin) * eased
        self._draw_slider()
        if self._animation_enabled and self.canvas.winfo_ismapped() and (elapsed < 1 or self._index >= 3):
            delay = self._FRAME_MS if elapsed < 1 else 80
            self._animation_id = self.after(delay, self._tick)

    def _geometry(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        thumb_radius = height * 13 / 34
        start = 1 + thumb_radius
        travel = max(1, width - 2 * start)
        return height / 2, thumb_radius, start, travel

    def _mouse_press(self, event):
        self.canvas.focus_set()
        return self._select(round(self._position_at(event.x)))

    def _mouse_drag(self, event):
        position = self._position_at(event.x)
        index = round(position)
        threshold = _DRAG_THRESHOLDS.get(index)
        if self._animation_enabled and threshold is not None and index > self._index and position < threshold:
            if self._index < index - 1:
                self._select(index - 1)
            self._drag_gate = index
            self._animate_to(position - 0.06)
            return
        if index != self._index:
            self._select(index)
        else:
            self._release_drag()

    def _release_drag(self, _event=None):
        if self._drag_gate is not None:
            self._drag_gate = None
            self._animate_to(self._index)

    def _position_at(self, x):
        _, _, start, travel = self._geometry()
        return min(self._last_index, max(0, (x - start) * self._last_index / travel))

    def _select(self, index):
        self._command(min(self._last_index, max(0, index)))
        return "break"

    def _draw_slider(self, _event=None):
        canvas = self.canvas
        canvas.delete("all")
        if canvas.winfo_height() <= 1 or canvas.winfo_width() <= 1:
            return
        center_y, radius, start, travel = self._geometry()
        right = canvas.winfo_width() - 1
        track_radius = radius * 12 / 13
        progress = self._position / self._last_index
        fill_edge = 1 + (right - 1) * progress

        outline = self._track_outline(right, center_y, track_radius)
        canvas.create_polygon(outline, fill="#40404d", tags="preset-track")
        if progress > 0:
            clipped = [(min(x, fill_edge), y) for x, y in outline]
            colors = ("#3b82f6", "#3b82f6", "#6366f1", "#514b91", "#236c3b", "#a9542e")
            fill = colors[min(self._index, 5)] if self._animation_enabled else colors[0]
            canvas.create_polygon(clipped, fill=fill, tags="preset-fill")
            if self._animation_enabled and self._index >= 3:
                self._draw_texture(fill_edge, center_y, track_radius)
                self._draw_atmosphere(start, fill_edge, center_y, track_radius)
        marker_radius = radius * 2 / 13
        for index in range(self._last_index + 1):
            x = start + travel * index / self._last_index
            canvas.create_oval(x - marker_radius, center_y - marker_radius, x + marker_radius,
                               center_y + marker_radius, fill="#858596", width=0, tags="preset-marker")

        x = start + travel * progress
        glow_tier = self._drag_gate or self._index
        if self._animation_enabled and glow_tier >= 3:
            glow = _GLOW_COLORS[min(glow_tier, 5) - 3]
            pulse = 1.2 + 0.6 * sin(monotonic() * 2)
            canvas.create_oval(x-radius-pulse, center_y-radius-pulse, x+radius+pulse, center_y+radius+pulse,
                               outline=glow, width=3, tags="preset-halo")
            age = monotonic() - self._arrival_started
            if self._drag_gate is None and 0 <= age < 0.5:
                ring = radius + 2 + age * 24
                canvas.create_oval(x-ring, center_y-ring, x+ring, center_y+ring,
                                   outline=glow, width=2, tags="preset-arrival")
        thumb_colors = ("#ffffff", "#ffffff", "#ffffff", "#e2e5ff", "#d6ffdf", "#ffe2c1")
        canvas.create_oval(x - radius, center_y - radius, x + radius, center_y + radius,
                           fill=thumb_colors[min(self._index, 5)] if self._animation_enabled else thumb_colors[0],
                           width=0, tags="preset-thumb")

    def _draw_texture(self, edge, center_y, radius):
        """Cache blurred, deterministic tier textures; crop only when the fill moves."""
        width = max(1, self.canvas.winfo_width() - 2)
        height = max(1, round(radius * 2))
        tier = min(self._index, 5)
        key = (tier, width, height)
        if key not in self._texture_cache:
            base, clouds = {
                3: ((53, 51, 107), ((93, 102, 191), (135, 112, 197), (111, 154, 220))),
                4: ((15, 71, 37), ((37, 143, 72), (100, 204, 118), (43, 116, 90))),
                5: ((91, 39, 42), ((195, 85, 44), (221, 138, 62), (130, 70, 111))),
            }[tier]
            rng = Random(20261123 + tier)
            image = Image.new("RGB", (width, height), base)
            blobs = Image.new("RGBA", (width, height))
            draw = ImageDraw.Draw(blobs)
            for i in range(34):
                x, y = rng.randrange(width), rng.randrange(height)
                rx, ry = rng.randint(12, 45), rng.randint(3, 9)
                draw.ellipse((x-rx, y-ry, x+rx, y+ry), fill=(*clouds[i % 3], rng.randint(55, 140)))
            image = Image.alpha_composite(image.convert("RGBA"), blobs.filter(ImageFilter.GaussianBlur(5)))
            draw = ImageDraw.Draw(image)
            for _ in range(max(25, width // 5)):
                x, y = rng.randrange(width), rng.randrange(height)
                draw.point((x, y), fill=(*clouds[2], rng.randint(140, 230)))
            mask = Image.new("L", (width, height))
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=height // 2, fill=255)
            image.putalpha(mask)
            self._texture_cache = {
                cached_key: cached_image for cached_key, cached_image in self._texture_cache.items()
                if cached_key[1:] == key[1:]
            }
            self._texture_cache[key] = image
        visible = max(0, min(width, round(edge - 1)))
        photo_key = (key, visible)
        if not visible:
            return
        if photo_key != self._texture_photo_key:
            image = self._texture_cache[key].crop((0, 0, visible, height))
            self._texture_photo = ImageTk.PhotoImage(image, master=self.canvas)
            self._texture_photo_key = photo_key
        self.canvas.create_image(1, center_y-height/2, image=self._texture_photo, anchor="nw", tags="preset-texture")

    def _draw_atmosphere(self, start, edge, center_y, radius):
        """Draw a small animated texture inside the straight part of the track."""
        canvas = self.canvas
        left, right = start + 2, edge - radius - 2
        if right <= left:
            return
        t = monotonic()
        span = right - left
        palette = {
            3: ("#706bc0", "#a6b7ff", "#d4dcff"),
            4: ("#43a563", "#88eaa1", "#d4ffdf"),
            5: ("#d87a3e", "#ffc47a", "#fff0c9"),
        }[min(self._index, 5)]
        # Narrow ribbons stay inside the rounded track and suggest the tier's atmosphere.
        for row in range(2):
            points = []
            for step in range(25):
                x = left + span * step / 24
                y = center_y + (row * 7 - 3.5) + 2 * sin(step * 0.48 + t * 0.8 + row)
                points.extend((x, y))
            canvas.create_line(*points, fill=palette[row], width=2, smooth=True, tags="preset-atmosphere")
        for i in range(18):
            x = left + span * ((i * 0.618 + t * 0.012) % 1)
            y = center_y + (i * 7 % 15) - 7
            size = 1.1 if sin(t * 2 + i) > 0 else 0.6
            canvas.create_oval(x-size, y-size, x+size, y+size, fill=palette[2], width=0,
                               tags="preset-atmosphere")
        if self._index == 4:
            x = left + span * 0.72
            canvas.create_polygon(x, center_y-6, x+4, center_y, x, center_y+6, x-4, center_y,
                                  fill=palette[1], outline=palette[2], tags="preset-atmosphere")
        elif self._index >= 5 and (t - self._arrival_started < 0.8 or sin(t * 2.1) > 0.88):
            x = left + span * 0.78
            canvas.create_line(x, center_y-7, x-4, center_y, x, center_y-1, x-3, center_y+7,
                               fill=palette[2], width=2, tags="preset-atmosphere")

    @staticmethod
    def _track_outline(right, center_y, radius):
        return [
            (center_x + radius * cos(radians(angle)), center_y + radius * sin(radians(angle)))
            for center_x, start in ((right - radius, -90), (1 + radius, 90))
            for angle in range(start, start + 181, 10)
        ]
