"""
BLE/USB HID Output

Sends recognized gestures as keyboard and mouse inputs using:
- USB HID (via pyautogui or pynput)
- BLE HID (via bled112 or bleak for BLE keyboard emulation)

Usage:
    controller = HIDController(mode="usb")
    controller.gesture_to_action("thumbs_up", {"velocity": ...})
"""

import time
import threading
from typing import Dict, Any, Optional, Callable

import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# Gesture-to-action mapping
GESTURE_ACTIONS: Dict[str, Dict[str, Any]] = {
    # Mouse actions
    "point": {"type": "mouse_move", "description": "Move cursor with index finger"},
    "open_hand": {"type": "mouse_move", "description": "Move cursor with open hand"},
    "pinch": {"type": "mouse_click", "button": "left", "description": "Left click"},
    "peace": {"type": "mouse_click", "button": "right", "description": "Right click"},
    "thumbs_up": {"type": "scroll", "direction": "up", "amount": 3, "description": "Scroll up"},
    "thumbs_down": {"type": "scroll", "direction": "down", "amount": 3, "description": "Scroll down"},

    # Keyboard actions
    "fist": {"type": "key_press", "key": "space", "description": "Space bar"},
    "swipe_right": {"type": "key_press", "key": "right", "description": "Arrow right"},
    "swipe_left": {"type": "key_press", "key": "left", "description": "Arrow left"},
    "wave": {"type": "key_press", "key": "escape", "description": "Escape key"},
    "push": {"type": "key_press", "key": "enter", "description": "Enter key"},
    "circle": {"type": "hotkey", "keys": ["cmd", "z"], "description": "Undo (Cmd+Z)"},
}


class HIDController:
    """
    Controls keyboard and mouse via HID emulation.

    Supports both USB (pyautogui/pynput) and BLE modes.
    """

    def __init__(
        self,
        mode: str = "usb",
        sensitivity: float = 1.0,
        dead_zone: float = 0.1,
    ):
        """
        Args:
            mode: "usb" or "ble"
            sensitivity: Mouse movement sensitivity multiplier
            dead_zone: Minimum movement threshold to trigger action
        """
        self.mode = mode
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone

        self._mouse = None
        self._keyboard = None
        self._enabled = False
        self._lock = threading.Lock()

        # Gesture action mapping
        self.action_map = dict(GESTURE_ACTIONS)

        # State
        self._last_cursor_pos = (0.5, 0.5)
        self._gesture_callbacks: Dict[str, Callable] = {}

        self._init_controllers()

    def _init_controllers(self):
        """Initialize HID controllers based on mode."""
        if self.mode == "usb":
            if PYNPUT_AVAILABLE:
                self._mouse = MouseController()
                self._keyboard = KeyboardController()
                logger.info("HID controller initialized (pynput)")
            elif PYAUTOGUI_AVAILABLE:
                logger.info("HID controller initialized (pyautogui)")
            else:
                logger.warning("No HID library available. Install pynput or pyautogui.")
                return
        elif self.mode == "ble":
            logger.info("BLE HID mode - requires external BLE keyboard firmware")
            return

        self._enabled = True

    @property
    def is_available(self) -> bool:
        return self._enabled

    def enable(self):
        """Enable HID output."""
        if self._mouse or PYAUTOGUI_AVAILABLE:
            self._enabled = True

    def disable(self):
        """Disable HID output."""
        self._enabled = False

    def gesture_to_action(
        self,
        gesture: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Execute the action mapped to a gesture.

        Args:
            gesture: Gesture name
            context: Additional context (velocity, landmarks, etc.)

        Returns:
            True if action was executed
        """
        if not self._enabled or gesture not in self.action_map:
            return False

        action = self.action_map[gesture]
        action_type = action.get("type")
        context = context or {}

        with self._lock:
            try:
                if action_type == "mouse_move":
                    return self._handle_mouse_move(context)
                elif action_type == "mouse_click":
                    return self._handle_mouse_click(action, context)
                elif action_type == "scroll":
                    return self._handle_scroll(action, context)
                elif action_type == "key_press":
                    return self._handle_key_press(action)
                elif action_type == "hotkey":
                    return self._handle_hotkey(action)
                else:
                    logger.warning("Unknown action type: %s", action_type)
                    return False
            except Exception as e:
                logger.error("HID action failed: %s", e)
                return False

    def _handle_mouse_move(self, context: Dict[str, Any]) -> bool:
        """Move mouse cursor based on hand position."""
        landmarks = context.get("landmarks", [])
        if not landmarks or len(landmarks) < 42:
            return False

        # Use index finger tip (landmark 8) position
        x = landmarks[8 * 2] / 640.0
        y = landmarks[8 * 2 + 1] / 480.0

        # Dead zone
        dx = x - self._last_cursor_pos[0]
        dy = y - self._last_cursor_pos[1]
        if abs(dx) < self.dead_zone and abs(dy) < self.dead_zone:
            return False

        self._last_cursor_pos = (x, y)

        if PYNPUT_AVAILABLE and self._mouse:
            import pyautogui
            screen_w, screen_h = pyautogui.size()
            self._mouse.position = (int(x * screen_w), int(y * screen_h))
        elif PYAUTOGUI_AVAILABLE:
            import pyautogui
            screen_w, screen_h = pyautogui.size()
            pyautogui.moveTo(int(x * screen_w), int(y * screen_h), _pause=False)

        return True

    def _handle_mouse_click(self, action: Dict, context: Dict) -> bool:
        """Perform mouse click."""
        button_str = action.get("button", "left")

        if PYNPUT_AVAILABLE and self._mouse:
            button = Button.left if button_str == "left" else Button.right
            self._mouse.click(button)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.click(_pause=False)

        return True

    def _handle_scroll(self, action: Dict, context: Dict) -> bool:
        """Perform scroll."""
        direction = action.get("direction", "up")
        amount = action.get("amount", 3)
        scroll_y = amount if direction == "up" else -amount

        if PYNPUT_AVAILABLE and self._mouse:
            self._mouse.scroll(0, scroll_y)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.scroll(scroll_y, _pause=False)

        return True

    def _handle_key_press(self, action: Dict) -> bool:
        """Press a single key."""
        key_str = action.get("key", "space")

        if PYNPUT_AVAILABLE and self._keyboard:
            key_map = {
                "space": Key.space,
                "enter": Key.enter,
                "escape": Key.esc,
                "left": Key.left,
                "right": Key.right,
                "up": Key.up,
                "down": Key.down,
            }
            key = key_map.get(key_str, Key.space)
            self._keyboard.press(key)
            self._keyboard.release(key)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.press(key_str, _pause=False)

        return True

    def _handle_hotkey(self, action: Dict) -> bool:
        """Press a key combination."""
        keys = action.get("keys", [])

        if PYNPUT_AVAILABLE and self._keyboard:
            key_map = {
                "cmd": Key.cmd,
                "ctrl": Key.ctrl,
                "alt": Key.alt,
                "shift": Key.shift,
                "z": "z",
                "c": "c",
                "v": "v",
            }
            pressed = [key_map.get(k, k) for k in keys]
            for k in pressed:
                self._keyboard.press(k)
            for k in reversed(pressed):
                self._keyboard.release(k)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey(*keys, _pause=False)

        return True

    def register_custom_action(self, gesture: str, action: Dict[str, Any]):
        """Register a custom gesture-to-action mapping."""
        self.action_map[gesture] = action

    def get_action_map(self) -> Dict[str, str]:
        """Get human-readable action descriptions."""
        return {k: v.get("description", v.get("type", "unknown")) for k, v in self.action_map.items()}
