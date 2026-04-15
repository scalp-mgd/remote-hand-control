"""Text inserter — pastes text at cursor via clipboard (supports Cyrillic).

Cross-platform: uses Win32 API on Windows, pyautogui on macOS/Linux.
"""

import platform
import time
import logging

import pyperclip

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


def _paste_windows():
    """Paste via Win32 keybd_event — reliable from background threads."""
    import ctypes
    VK_CONTROL = 0x11
    VK_V = 0x56
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_V, 0, 0, 0)
    time.sleep(0.02)
    ctypes.windll.user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _paste_mac():
    """Paste via Cmd+V on macOS."""
    import pyautogui
    pyautogui.hotkey("command", "v", _pause=False)


def _paste_linux():
    """Paste via Ctrl+V on Linux."""
    import pyautogui
    pyautogui.hotkey("ctrl", "v", _pause=False)


def insert_text(text: str):
    """Copy text to clipboard and paste.

    This is the only reliable way to input Cyrillic/CJK text cross-platform.
    """
    if not text:
        return
    pyperclip.copy(text)
    time.sleep(0.05)  # small delay for clipboard reliability

    if _SYSTEM == "Windows":
        _paste_windows()
    elif _SYSTEM == "Darwin":
        _paste_mac()
    else:
        _paste_linux()

    logger.debug("Pasted: %s", text[:50])
