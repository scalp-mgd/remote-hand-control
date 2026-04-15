"""State machine — dispatches gestures to actions with cooldown and debounce."""

import enum
import logging
import math
import time
from typing import Optional

import pyautogui

from core.cursor_controller import CursorController
from core.voice_recorder import RealtimeVoiceRecorder

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    RECORDING = "RECORDING"


class StateMachine:
    def __init__(
        self,
        cursor: CursorController,
        recorder: RealtimeVoiceRecorder,
        action_mapping: dict[str, str] | None = None,
        cooldown_frames: int = 15,
        debounce_frames: int = 3,
        confidence_threshold: float = 0.8,
        pinch_threshold: float = 0.06,
    ):
        self.state = State.IDLE
        self._cursor = cursor
        self._recorder = recorder

        self._mapping = action_mapping or {
            "open_palm": "move_cursor",
            "fist": "enter",
            "one_finger_up": "toggle_recording",
        }

        self._cooldown_frames = cooldown_frames
        self._debounce_frames = debounce_frames
        self._confidence_threshold = confidence_threshold

        # Debounce
        self._last_gesture: str = ""
        self._gesture_count: int = 0
        self._action_fired_for_gesture: bool = False

        # Grace period for TRACKING
        self._tracking_grace_frames: int = 15
        self._grace_counter: int = 0

        # Cooldown
        self._cooldown_counters: dict[str, int] = {}

        # Pinch detection (thumb + index = left click / scroll)
        self._pinch_threshold: float = pinch_threshold
        self._pinch_active: bool = False
        self._pinch_was_active: bool = False  # edge detection
        self._pinch_start_time: float = 0.0   # when pinch started
        self._pinch_hold_sec: float = 0.3     # hold this long = scroll mode
        self._pinch_scrolling: bool = False    # currently scrolling?
        self._pinch_start_y: float = 0.0      # hand Y when pinch started
        self._scroll_sensitivity: float = 1500.0 # scroll speed multiplier
        self._last_click_time: float = 0.0    # for double-click detection
        self._dblclick_window: float = 0.7    # max seconds between clicks for dbl

        # Right-click detection (thumb + middle finger)
        self._rclick_active: bool = False
        self._rclick_was_active: bool = False

        # Drag detection (thumb + index + ring = 3 fingers)
        self._drag_active: bool = False
        self._drag_was_active: bool = False
        self._dragging: bool = False  # mouse button held down

        # Track when hand was last seen
        self._hand_lost_time: float | None = None

        # HUD
        self.status_message: str = ""
        self.last_action: str = ""          # shown on HUD overlay
        self._action_display_time: float = 0.0  # when last action was set
        self._action_display_duration: float = 1.0  # show for 1 sec

    def _set_action(self, action_name: str, now: float = 0.0):
        """Set current action for HUD display (auto-clears after duration)."""
        self.last_action = action_name
        self._action_display_time = now or time.time()

    def _finger_distance(self, landmarks_norm, i, j) -> float:
        """Distance between two landmarks (normalized coords)."""
        a, b = landmarks_norm[i], landmarks_norm[j]
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _detect_pinch(self, landmarks_norm: list[tuple[float, float]]) -> bool:
        """Detect pinch gesture: thumb tip (4) close to index tip (8)."""
        return self._finger_distance(landmarks_norm, 4, 8) < self._pinch_threshold

    def _detect_right_click(self, landmarks_norm: list[tuple[float, float]]) -> bool:
        """Detect right-click gesture: thumb tip (4) close to middle tip (12)."""
        return self._finger_distance(landmarks_norm, 4, 12) < self._pinch_threshold

    def _detect_drag(self, landmarks_norm: list[tuple[float, float]]) -> bool:
        """Detect drag gesture: thumb tip (4) close to ring tip (16)."""
        return self._finger_distance(landmarks_norm, 4, 16) < self._pinch_threshold

    def update(
        self,
        gesture: Optional[tuple[int, str, float]],
        landmarks_norm: Optional[list[tuple[float, float]]],
    ):
        """Process one frame."""
        now = time.time()

        # Auto-clear action display after duration
        if self.last_action and now - self._action_display_time > self._action_display_duration:
            self.last_action = ""

        for key in self._cooldown_counters:
            self._cooldown_counters[key] += 1

        if landmarks_norm is None:
            self._handle_no_hand()
            return

        self._hand_lost_time = None

        # Gesture label from classifier
        if gesture is not None:
            _, label, confidence = gesture
            if confidence < self._confidence_threshold:
                label = "uncertain"
        else:
            label = "uncertain"

        # --- Gesture detection ONLY during open_palm (not fist!) ---
        if label == "open_palm":
            self._pinch_active = self._detect_pinch(landmarks_norm)
            self._rclick_active = self._detect_right_click(landmarks_norm)
            self._drag_active = self._detect_drag(landmarks_norm)
            # If pinch and rclick both active, prefer pinch
            if self._pinch_active and self._rclick_active:
                self._rclick_active = False
            # If dragging, suppress pinch (different fingers but avoid confusion)
            if self._drag_active and self._pinch_active:
                self._pinch_active = False
        else:
            self._pinch_active = False
            self._rclick_active = False
            self._drag_active = False

        # Debounce
        if label == self._last_gesture:
            self._gesture_count += 1
        else:
            self._last_gesture = label
            self._gesture_count = 1
            self._action_fired_for_gesture = False

        debounced = self._gesture_count >= self._debounce_frames
        action = self._mapping.get(label)

        # === PINCH: short = click, long + move = scroll ===
        if self._pinch_active:
            hand_y = landmarks_norm[8][1]  # index tip Y (0=top, 1=bottom)

            if not self._pinch_was_active:
                # Pinch just started
                self._pinch_start_time = now
                self._pinch_start_y = hand_y
                self._pinch_scrolling = False
                self._pinch_was_active = True
            else:
                # Pinch held — check if should scroll
                hold_duration = now - self._pinch_start_time
                if hold_duration >= self._pinch_hold_sec:
                    # Scroll mode: hand Y delta → scroll
                    if not self._pinch_scrolling:
                        self._pinch_scrolling = True
                        self._pinch_start_y = hand_y  # reset baseline
                        self._set_action("SCROLL")
                        logger.info("Action: scroll mode")

                    dy = (hand_y - self._pinch_start_y) * self._scroll_sensitivity
                    if abs(dy) > 0.05:
                        scroll_amount = int(dy)
                        if scroll_amount != 0:
                            pyautogui.scroll(scroll_amount, _pause=False)
                            self._pinch_start_y = hand_y  # consume delta

        elif self._pinch_was_active:
            # Pinch just released
            if not self._pinch_scrolling:
                # Was short pinch → click or double-click
                if self.state == State.TRACKING:
                    if now - self._last_click_time < self._dblclick_window:
                        # Double click!
                        pyautogui.doubleClick(_pause=False)
                        self._set_action("DBL-CLICK")
                        self._last_click_time = 0.0  # reset so triple doesn't fire
                        logger.info("Action: double click")
                    else:
                        # Single click
                        self._cursor.click()
                        self._set_action("CLICK")
                        self._last_click_time = now
                        logger.info("Action: pinch click")
            else:
                self._set_action("SCROLL END")
                logger.info("Action: scroll end")
            self._pinch_was_active = False
            self._pinch_scrolling = False

        # === RIGHT CLICK: thumb + middle finger (edge-triggered) ===
        if self._rclick_active and not self._rclick_was_active:
            if self.state == State.TRACKING and self._check_cooldown("rclick"):
                self._cursor.right_click()
                self._reset_cooldown("rclick")
                self._set_action("R-CLICK")
                logger.info("Action: right click")
            self._rclick_was_active = True
        elif not self._rclick_active:
            self._rclick_was_active = False

        # === DRAG: thumb + index + ring (hold mouse + move) ===
        if self._drag_active:
            if not self._drag_was_active:
                # Drag started — press mouse button down
                if self.state == State.TRACKING:
                    pyautogui.mouseDown(_pause=False)
                    self._dragging = True
                    self._set_action("DRAG")
                    logger.info("Action: drag start")
                self._drag_was_active = True
            else:
                # Dragging — move cursor with hand
                index_tip = landmarks_norm[8]
                self._cursor.update(index_tip)
        elif self._drag_was_active:
            # Drag released — release mouse button
            if self._dragging:
                pyautogui.mouseUp(_pause=False)
                self._dragging = False
                self._cursor.release()  # reset delta tracking
                self._set_action("DROP")
                logger.info("Action: drop")
            self._drag_was_active = False

        # === OPEN PALM: move cursor (only when NOT pinching/rclicking/dragging) ===
        if action == "move_cursor" and debounced and landmarks_norm and not self._pinch_active and not self._rclick_active and not self._drag_active:
            index_tip = landmarks_norm[8]
            self._cursor.update(index_tip)
            self._grace_counter = 0

            if self.state == State.IDLE and not self._action_fired_for_gesture:
                self.state = State.TRACKING
                self._action_fired_for_gesture = True
                self._set_action("MOVE")
                logger.info("State: IDLE -> TRACKING")
            return

        # === FIST: press Enter (only in TRACKING state) ===
        if action == "enter" and debounced and not self._action_fired_for_gesture:
            if self.state == State.TRACKING and self._check_cooldown("enter"):
                pyautogui.press("enter", _pause=False)
                self._reset_cooldown("enter")
                self._action_fired_for_gesture = True
                self._set_action("ENTER")
                logger.info("Action: Enter (fist)")
            return

        # === FINGER UP: HOLD to record (real-time STT) ===
        if action == "toggle_recording" and debounced:
            if self.state != State.RECORDING:
                self.state = State.RECORDING
                self._recorder.start()
                self.status_message = ""
                self._action_fired_for_gesture = True
                self._set_action("REC START")
                logger.info("State: -> RECORDING (real-time STT)")
            return  # keep recording while finger is up

        # === Other gesture: stop recording if active ===
        if self.state == State.RECORDING and label != "one_finger_up":
            self._recorder.stop()
            self.state = State.IDLE
            self.status_message = "Done"
            self._set_action("REC STOP")
            logger.info("State: RECORDING -> IDLE")
            return

        # === Grace period for TRACKING ===
        if self.state == State.TRACKING and label in ("uncertain", "other"):
            self._cursor.release()
            self._grace_counter += 1
            if self._grace_counter > self._tracking_grace_frames:
                self.state = State.IDLE
                self._grace_counter = 0

    def _handle_no_hand(self):
        self._last_gesture = ""
        self._gesture_count = 0
        self._action_fired_for_gesture = False
        self._grace_counter = 0
        self._pinch_was_active = False
        self._pinch_scrolling = False
        self._pinch_active = False
        self._rclick_active = False
        self._rclick_was_active = False
        self._drag_active = False
        self._drag_was_active = False
        if self._dragging:
            pyautogui.mouseUp(_pause=False)
            self._dragging = False
        self._cursor.release()

        if self.state == State.TRACKING:
            self.state = State.IDLE

        elif self.state == State.RECORDING:
            if self._hand_lost_time is None:
                self._hand_lost_time = time.time()
            elif time.time() - self._hand_lost_time > 1.5:
                self._recorder.stop()
                self.state = State.IDLE
                self.status_message = "Done"
                logger.info("State: RECORDING -> IDLE (hand lost)")
                self._hand_lost_time = None

    def _check_cooldown(self, action: str) -> bool:
        return self._cooldown_counters.get(action, self._cooldown_frames) >= self._cooldown_frames

    def _reset_cooldown(self, action: str):
        self._cooldown_counters[action] = 0
