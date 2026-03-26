from config.settings import SCREEN_WIDTH, KEY_WIDTH, KEY_HEIGHT, KEY_PADDING

# QWERTY rows
ROWS = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'BKSP'],
    ['TAB', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['CAPS', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'ENTER'],
    ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'SHIFT'],
    ['SPACE']
]

# Special key widths as multiplier of KEY_WIDTH
SPECIAL_WIDTHS = {
    'BKSP':  2.0,
    'TAB':   1.5,
    'CAPS':  1.8,
    'ENTER': 2.2,
    'SHIFT': 2.5,
    'SPACE': 8.0,
}

# What each special key outputs or does
SPECIAL_ACTIONS = {
    'BKSP':  'backspace',
    'TAB':   'tab',
    'CAPS':  'caps',
    'ENTER': 'enter',
    'SHIFT': 'shift',
    'SPACE': 'space',
}

# Shifted character map
SHIFT_MAP = {
    '`': '~',  '1': '!',  '2': '@',  '3': '#',  '4': '$',
    '5': '%',  '6': '^',  '7': '&',  '8': '*',  '9': '(',
    '0': ')',  '-': '_',  '=': '+',  '[': '{',  ']': '}',
    '\\': '|', ';': ':',  "'": '"',  ',': '<',  '.': '>',
    '/': '?'
}


def build_key_rects(
    keyboard_y: int
) -> list[dict]:
    """
    Build list of key dicts with positions and sizes.
    Each key: {label, x, y, w, h, action}
    keyboard_y: top y position of keyboard on screen
    """
    keys = []
    y = keyboard_y

    for row in ROWS:
        # Calculate total row width to center it
        total_w = sum(
            int(SPECIAL_WIDTHS.get(k, 1.0) * KEY_WIDTH) + KEY_PADDING
            for k in row
        )
        x = (SCREEN_WIDTH - total_w) // 2

        for label in row:
            w = int(SPECIAL_WIDTHS.get(label, 1.0) * KEY_WIDTH)
            h = KEY_HEIGHT

            action = SPECIAL_ACTIONS.get(label, label.lower())

            keys.append({
                "label":  label,
                "x":      x,
                "y":      y,
                "w":      w,
                "h":      h,
                "action": action,
            })
            x += w + KEY_PADDING

        y += KEY_HEIGHT + KEY_PADDING

    return keys


def get_keyboard_height() -> int:
    """Total pixel height of the keyboard."""
    rows      = len(ROWS)
    return rows * (KEY_HEIGHT + KEY_PADDING) + KEY_PADDING


def get_keyboard_y() -> int:
    """Y position where keyboard starts (bottom of screen)."""
    from config.settings import SCREEN_HEIGHT
    return SCREEN_HEIGHT - get_keyboard_height() - 10