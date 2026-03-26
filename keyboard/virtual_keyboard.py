import pygame
import pyautogui
import os
import time
import cv2
import numpy as np
from utils.logger        import get_logger
from config.settings     import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    KEY_WIDTH, KEY_HEIGHT, KEY_PADDING,
    DWELL_TIME, DWELL_COOLDOWN,
    FRAME_WIDTH, FRAME_HEIGHT
)
from keyboard.key_layout     import build_key_rects, get_keyboard_y, get_keyboard_height
from keyboard.word_predictor import WordPredictor

logger = get_logger(__name__)

# Colors
C_BG         = (20,  20,  20)
C_KEY        = (50,  50,  50)
C_KEY_HOVER  = (80, 130, 200)
C_KEY_DWELL  = (40, 200, 100)
C_KEY_SPEC   = (70,  40,  40)
C_TEXT       = (240, 240, 240)
C_TEXT_BOX   = (30,  30,  30)
C_TEXT_INPUT = (255, 255, 255)
C_SUGGEST    = (60,  60,  100)
C_SUGGEST_HV = (100, 100, 180)
C_PROGRESS   = (40, 200, 100)
C_GAZE_DOT   = (0,  255,  0)
C_DWELL_RING = (0,  200, 255)

SPECIAL_KEYS        = {'BKSP', 'TAB', 'CAPS', 'ENTER', 'SHIFT', 'SPACE'}
SPECIAL_ACTIONS_SKIP = {'caps', 'shift', 'backspace', 'enter', 'tab', 'space'}
SUGGESTION_H  = 44
TEXT_BOX_H    = 48
TOP_PANEL_H   = SUGGESTION_H + TEXT_BOX_H

# Camera preview dimensions inside Pygame window
CAM_W = 240
CAM_H = 180


class VirtualKeyboard:
    def __init__(self, shared_state):
        self.state      = shared_state
        self.predictor  = WordPredictor()
        self.running    = False
        self.on_quit    = None

        self.typed_text = ""
        self.caps       = False
        self.shift      = False

        self.keyboard_y = get_keyboard_y()
        self.keys       = build_key_rects(self.keyboard_y + TOP_PANEL_H)

        self._dwell_start:  dict[int, float] = {}
        self._last_trigger: dict[int, float] = {}
        self._hover_idx:    int | None        = None

        logger.info("VirtualKeyboard initialized.")

    def start_main_thread(self, on_quit=None):
        self.on_quit = on_quit
        self.running = True
        self._run()

    def stop(self):
        self.running = False

    def _run(self):
        kb_h = get_keyboard_height() + TOP_PANEL_H + 20
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,{SCREEN_HEIGHT - kb_h}"

        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, kb_h), pygame.NOFRAME)
        pygame.display.set_caption("GazeControl")

        font_key  = pygame.font.SysFont("Arial", 16, bold=True)
        font_text = pygame.font.SysFont("Arial", 22)
        font_sug  = pygame.font.SysFont("Arial", 18)
        font_dbg  = pygame.font.SysFont("Arial", 13)

        clock = pygame.time.Clock()

        while self.running:
            # --- Drain gaze queue ---
            gaze_x, gaze_y = None, None
            while not self.state.gaze_queue.empty():
                try:
                    gaze_x, gaze_y = self.state.gaze_queue.get_nowait()
                except Exception:
                    break

            local_y = None
            if gaze_x is not None and gaze_y is not None:
                local_y = gaze_y - (SCREEN_HEIGHT - kb_h)

            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    if self.on_quit:
                        self.on_quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                        if self.on_quit:
                            self.on_quit()

            # --- Update dwell ---
            if gaze_x is not None and local_y is not None:
                self._update_dwell(gaze_x, local_y)

            # --- Draw ---
            screen.fill(C_BG)
            self._draw_text_box(screen, font_text)
            self._draw_suggestions(screen, font_sug)
            self._draw_keys(screen, font_key)
            self._draw_camera_preview(screen, font_dbg)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    # ------------------------------------------------------------------ #
    #  Camera preview (top right corner of keyboard window)
    # ------------------------------------------------------------------ #

    def _draw_camera_preview(self, screen, font):
        with self.state.lock:
            frame      = self.state.frame
            debug_info = self.state.debug_info
            dwell_prog = self.state.dwell_prog
            frame_gaze = self.state.frame_gaze
            flashing   = self.state.flashing

        if frame is None:
            return

        # Draw gaze dot on frame
        frame = frame.copy()
        fx, fy = frame_gaze
        cv2.circle(frame, (fx, fy), 6,  (0, 255, 0), -1)
        cv2.circle(frame, (fx, fy), 8,  (0, 255, 0), 1)
        if dwell_prog > 0.0:
            angle = int(360 * dwell_prog)
            cv2.ellipse(frame, (fx, fy), (20, 20), -90, 0, angle, (0, 200, 255), 2)

        if flashing:
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 255, 0), 6)

        # Resize and convert BGR→RGB for pygame
        small = cv2.resize(frame, (CAM_W, CAM_H))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf  = pygame.surfarray.make_surface(np.transpose(small, (1, 0, 2)))

        # Position: top right
        cam_x = SCREEN_WIDTH - CAM_W - 8
        cam_y = TOP_PANEL_H + 4
        screen.blit(surf, (cam_x, cam_y))

        # Border
        pygame.draw.rect(screen, (80, 80, 80),
                         (cam_x, cam_y, CAM_W, CAM_H), 1)

        # Debug info below camera
        if debug_info:
            dy = cam_y + CAM_H + 6
            for k, v in debug_info.items():
                line = f"{k}: {v}"
                surf_txt = font.render(line, True, (180, 180, 180))
                if cam_x + surf_txt.get_width() < SCREEN_WIDTH - 4:
                    screen.blit(surf_txt, (cam_x, dy))
                    dy += 16
                    if dy > cam_y + CAM_H + 16 * 8:
                        break

    # ------------------------------------------------------------------ #
    #  Dwell
    # ------------------------------------------------------------------ #

    def _update_dwell(self, gx: float, gy: float):
        now = time.time()
        hit = self._hit_test(gx, gy)

        if hit is None:
            self._dwell_start.clear()
            self._hover_idx = None
            return

        self._hover_idx = hit
        last = self._last_trigger.get(hit, 0.0)
        if now - last < DWELL_COOLDOWN:
            return

        if hit not in self._dwell_start:
            self._dwell_start = {hit: now}
            return

        elapsed = now - self._dwell_start[hit]
        if elapsed >= DWELL_TIME:
            self._fire_key(hit)
            self._last_trigger[hit] = now
            self._dwell_start.clear()

    def _hit_test(self, gx: float, gy: float) -> int | None:
        key_off = TOP_PANEL_H
        for i, key in enumerate(self.keys):
            ky = key["y"] - self.keyboard_y - TOP_PANEL_H + key_off
            if (key["x"] <= gx <= key["x"] + key["w"] and
                    ky <= gy <= ky + key["h"]):
                return i

        suggestions = self.predictor.predict(self.typed_text)
        if suggestions:
            sw  = SCREEN_WIDTH // 3
            sy0 = TEXT_BOX_H
            sy1 = TEXT_BOX_H + SUGGESTION_H
            for i, _ in enumerate(suggestions):
                sx0 = i * sw
                if sx0 <= gx <= sx0 + sw and sy0 <= gy <= sy1:
                    return -(i + 1)
        return None

    def _fire_key(self, idx: int):
        if idx < 0:
            suggestions = self.predictor.predict(self.typed_text)
            slot = -(idx + 1)
            if slot < len(suggestions):
                word  = suggestions[slot]
                parts = self.typed_text.split(" ")
                last  = parts[-1] if parts else ""
                remaining = word[len(last):]
                self.typed_text = " ".join(parts[:-1] + [word]) + " "
                pyautogui.typewrite(remaining + " ", interval=0.02)
            return

        key    = self.keys[idx]
        action = key["action"]

        if action == "caps":
            self.caps  = not self.caps
            self.shift = False
            return
        if action == "shift":
            self.shift = not self.shift
            return
        if action == "backspace":
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
            pyautogui.press("backspace")
            return
        if action in ("enter", "tab", "space"):
            char = {"enter": "\n", "tab": "\t", "space": " "}[action]
            self.typed_text += char
            pyautogui.press(action)
            return

        char = key["label"]
        if self.caps or self.shift:
            from keyboard.key_layout import SHIFT_MAP
            char = SHIFT_MAP.get(char, char.upper())
        else:
            char = char.lower()

        self.typed_text += char
        pyautogui.typewrite(char, interval=0.0)
        if self.shift:
            self.shift = False

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #

    def _draw_text_box(self, screen, font):
        pygame.draw.rect(screen, C_TEXT_BOX, (0, 0, SCREEN_WIDTH, TEXT_BOX_H))
        pygame.draw.line(screen, (80, 80, 80),
                         (0, TEXT_BOX_H), (SCREEN_WIDTH, TEXT_BOX_H), 1)
        display = self.typed_text[-60:]
        surf    = font.render(display + "|", True, C_TEXT_INPUT)
        screen.blit(surf, (12, (TEXT_BOX_H - surf.get_height()) // 2))

    def _draw_suggestions(self, screen, font):
        suggestions = self.predictor.predict(self.typed_text)
        if not suggestions:
            return

        sw  = SCREEN_WIDTH // 3
        y0  = TEXT_BOX_H
        now = time.time()

        for i, word in enumerate(suggestions):
            x0   = i * sw
            slot = -(i + 1)
            bg   = C_SUGGEST_HV if self._hover_idx == slot else C_SUGGEST
            pygame.draw.rect(screen, bg, (x0, y0, sw - 2, SUGGESTION_H))

            if slot in self._dwell_start:
                elapsed  = now - self._dwell_start[slot]
                progress = min(elapsed / DWELL_TIME, 1.0)
                pygame.draw.rect(screen, C_PROGRESS,
                                 (x0, y0 + SUGGESTION_H - 4,
                                  int(sw * progress), 4))

            surf = font.render(word, True, C_TEXT)
            screen.blit(surf, (x0 + 8, y0 + (SUGGESTION_H - surf.get_height()) // 2))

        pygame.draw.line(screen, (80, 80, 80),
                         (0, TEXT_BOX_H + SUGGESTION_H),
                         (SCREEN_WIDTH, TEXT_BOX_H + SUGGESTION_H), 1)

    def _draw_keys(self, screen, font):
        now     = time.time()
        key_off = TOP_PANEL_H

        for i, key in enumerate(self.keys):
            ky         = key["y"] - self.keyboard_y - TOP_PANEL_H + key_off
            is_special = key["label"] in SPECIAL_KEYS
            is_hover   = (self._hover_idx == i)

            progress = 0.0
            if i in self._dwell_start:
                elapsed  = now - self._dwell_start[i]
                progress = min(elapsed / DWELL_TIME, 1.0)

            if is_hover:
                color = C_KEY_DWELL if progress > 0.5 else C_KEY_HOVER
            elif is_special:
                color = C_KEY_SPEC
            else:
                color = C_KEY

            if key["action"] == "caps"  and self.caps:  color = (100, 180, 100)
            if key["action"] == "shift" and self.shift: color = (100, 180, 100)

            pygame.draw.rect(screen, color,
                             (key["x"], ky, key["w"], key["h"]),
                             border_radius=6)

            if progress > 0.0:
                pygame.draw.rect(screen, C_PROGRESS,
                                 (key["x"], ky + key["h"] - 4,
                                  int(key["w"] * progress), 4),
                                 border_radius=3)

            label = key["label"]
            if key["action"] not in SPECIAL_ACTIONS_SKIP:
                from keyboard.key_layout import SHIFT_MAP
                if self.caps or self.shift:
                    label = SHIFT_MAP.get(label, label.upper())
                else:
                    label = label.lower()

            surf = font.render(label, True, C_TEXT)
            tx   = key["x"] + (key["w"] - surf.get_width())  // 2
            ty   = ky        + (key["h"] - surf.get_height()) // 2
            screen.blit(surf, (tx, ty))