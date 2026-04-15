"""Cursor controller — RELATIVE (trackpad-style) hand-to-cursor mapping."""

import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


class CursorController:
    def __init__(
        self,
        smoothing_alpha: float = 0.12,
        sensitivity: float = 1.5,
    ):
        self._alpha = smoothing_alpha
        self._sensitivity = sensitivity
        self._screen_w, self._screen_h = pyautogui.size()

        # Previous hand position (normalized) — for computing deltas
        self._prev_nx: float | None = None
        self._prev_ny: float | None = None

        # Smoothed delta
        self._smooth_dx: float = 0.0
        self._smooth_dy: float = 0.0

    def update(self, landmark_norm: tuple[float, float]):
        """Move cursor RELATIVE to hand movement (trackpad-style).

        Hand position in camera frame doesn't matter — only MOVEMENT counts.
        Move hand right → cursor moves right. Move hand down → cursor moves down.
        """
        # Mirror X (webcam is mirrored)
        nx = 1.0 - landmark_norm[0]
        ny = landmark_norm[1]

        if self._prev_nx is None:
            # First frame after tracking starts — just save position, don't move
            self._prev_nx = nx
            self._prev_ny = ny
            return

        # Raw delta in normalized coords
        raw_dx = (nx - self._prev_nx) * self._screen_w * self._sensitivity
        raw_dy = (ny - self._prev_ny) * self._screen_h * self._sensitivity

        self._prev_nx = nx
        self._prev_ny = ny

        # EMA smooth the delta to reduce jitter
        self._smooth_dx = self._alpha * raw_dx + (1 - self._alpha) * self._smooth_dx
        self._smooth_dy = self._alpha * raw_dy + (1 - self._alpha) * self._smooth_dy

        # Skip tiny movements (deadzone to prevent drift)
        if abs(self._smooth_dx) < 0.5 and abs(self._smooth_dy) < 0.5:
            return

        # Get current position and apply delta
        cx, cy = pyautogui.position()
        new_x = max(0, min(self._screen_w - 1, int(cx + self._smooth_dx)))
        new_y = max(0, min(self._screen_h - 1, int(cy + self._smooth_dy)))

        pyautogui.moveTo(new_x, new_y, _pause=False)

    def click(self):
        pyautogui.click(_pause=False)

    def right_click(self):
        pyautogui.click(button='right', _pause=False)

    def release(self):
        """Call when hand is lost or gesture changes — reset delta tracking."""
        self._prev_nx = None
        self._prev_ny = None
        self._smooth_dx = 0.0
        self._smooth_dy = 0.0
