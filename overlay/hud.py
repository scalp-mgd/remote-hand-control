"""HUD overlay — draws landmarks, gesture label, FPS, recording status on frame."""

import cv2
import numpy as np

# MediaPipe hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

# Help table content (English only — cv2 can't render Cyrillic)
HELP_ROWS = [
    ("Open palm",           "Move cursor"),
    ("Quick pinch",         "Click"),
    ("Double pinch",        "Double click"),
    ("Long pinch + move",   "Scroll"),
    ("Thumb + ring",        "Drag & drop"),
    ("Thumb + middle",      "Right click"),
    ("Fist",                "Enter"),
    ("Finger up (hold)",    "Voice input"),
]

# Help button position (top-left area, below state)
HELP_BTN_X = 10
HELP_BTN_Y = 80
HELP_BTN_R = 14


class HelpTooltip:
    """Tracks mouse hover state for the help '?' button."""

    def __init__(self):
        self.hovered = False
        self._registered = False

    def register(self, window_name: str):
        """Register mouse callback on the OpenCV window."""
        if not self._registered:
            cv2.setMouseCallback(window_name, self._on_mouse)
            self._registered = True

    def _on_mouse(self, event, x, y, flags, param):
        dx = x - HELP_BTN_X - HELP_BTN_R
        dy = y - HELP_BTN_Y
        self.hovered = (dx * dx + dy * dy) <= (HELP_BTN_R + 10) ** 2


# Global instance
help_tooltip = HelpTooltip()


def draw_hud(
    frame: np.ndarray,
    landmarks: list[tuple[float, float]] | None = None,
    gesture_label: str = "",
    confidence: float = 0.0,
    fps: float = 0.0,
    is_recording: bool = False,
    state: str = "IDLE",
    show_landmarks: bool = True,
    status_message: str = "",
    partial_text: str = "",
    last_action: str = "",
    is_transcribing: bool = False,
) -> np.ndarray:
    """Draw all HUD elements on the frame (mutates in place and returns it)."""
    h, w = frame.shape[:2]

    # Draw landmarks and skeleton
    if landmarks and show_landmarks:
        for connection in HAND_CONNECTIONS:
            p1 = (int(landmarks[connection[0]][0]), int(landmarks[connection[0]][1]))
            p2 = (int(landmarks[connection[1]][0]), int(landmarks[connection[1]][1]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        for i, (x, y) in enumerate(landmarks):
            color = (0, 0, 255) if i == 8 else (255, 0, 0)  # index tip = red
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)

    # FPS (top-left)
    cv2.putText(
        frame, f"FPS: {fps:.0f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    # State (top-left, below FPS)
    cv2.putText(
        frame, f"State: {state}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    # Help "?" button (below state)
    btn_center = (HELP_BTN_X + HELP_BTN_R, HELP_BTN_Y)
    btn_color = (255, 255, 0) if help_tooltip.hovered else (180, 180, 180)
    cv2.circle(frame, btn_center, HELP_BTN_R, btn_color, 2)
    cv2.putText(
        frame, "?", (HELP_BTN_X + 7, HELP_BTN_Y + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, btn_color, 2,
    )

    # Help table overlay (on hover)
    if help_tooltip.hovered:
        _draw_help_table(frame, w, h)

    # Gesture label + confidence (top-right)
    if gesture_label:
        text = f"{gesture_label} ({confidence:.0%})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(
            frame, text, (w - text_size[0] - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

    # Recording indicator (top-center, red dot + "REC")
    if is_recording:
        center_x = w // 2
        cv2.circle(frame, (center_x - 30, 25), 10, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (center_x - 15, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
    elif is_transcribing:
        # Yellow indicator while waiting for Groq response
        center_x = w // 2
        cv2.circle(frame, (center_x - 50, 25), 10, (0, 200, 255), -1)
        cv2.putText(
            frame, "Transcribing...", (center_x - 35, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )


    # Current action (bottom-center, large, color-coded)
    if last_action:
        action_colors = {
            "CLICK": (0, 200, 0),       # green
            "DBL-CLICK": (0, 255, 100),  # bright green
            "R-CLICK": (200, 0, 200),   # purple
            "DRAG": (0, 165, 255),      # orange
            "DROP": (0, 100, 200),      # dark orange
            "ENTER": (0, 140, 255),      # orange
            "MOVE": (255, 200, 0),       # cyan-ish
            "SCROLL": (255, 100, 0),     # blue
            "SCROLL END": (200, 100, 0), # dark blue
            "REC START": (0, 0, 255),    # red
            "REC STOP": (100, 100, 255), # light red
        }
        color = action_colors.get(last_action, (255, 255, 255))
        text_size = cv2.getTextSize(last_action, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        x = (w - text_size[0]) // 2
        y = h - 30
        # Dark background for readability
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(
            frame, last_action, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3,
        )
    elif status_message:
        # Status message (bottom-center) — only if no action showing
        text_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame, status_message, ((w - text_size[0]) // 2, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )

    return frame


def _draw_help_table(frame: np.ndarray, w: int, h: int):
    """Draw the gesture help table overlay."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    row_h = 24
    col1_w = 200
    col2_w = 140
    table_w = col1_w + col2_w + 20
    table_h = (len(HELP_ROWS) + 1) * row_h + 16  # +1 for header

    # Position: anchored below the "?" button
    tx = HELP_BTN_X
    ty = HELP_BTN_Y + HELP_BTN_R + 8

    # Clamp to frame
    if tx + table_w > w:
        tx = w - table_w - 5
    if ty + table_h > h:
        ty = h - table_h - 5

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx, ty), (tx + table_w, ty + table_h), (30, 30, 30), -1)
    cv2.rectangle(overlay, (tx, ty), (tx + table_w, ty + table_h), (100, 100, 100), 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Header
    y = ty + 18
    cv2.putText(frame, "Gesture", (tx + 10, y), font, font_scale, (0, 255, 255), thickness)
    cv2.putText(frame, "Action", (tx + col1_w + 10, y), font, font_scale, (0, 255, 255), thickness)
    y += 4
    cv2.line(frame, (tx + 5, y), (tx + table_w - 5, y), (100, 100, 100), 1)

    # Rows
    for gesture, action in HELP_ROWS:
        y += row_h
        cv2.putText(frame, gesture, (tx + 10, y), font, font_scale, (220, 220, 220), thickness)
        cv2.putText(frame, action, (tx + col1_w + 10, y), font, font_scale, (150, 255, 150), thickness)
