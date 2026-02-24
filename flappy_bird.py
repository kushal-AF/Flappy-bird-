from __future__ import annotations

import random
import sys
from dataclasses import dataclass

import pygame

# Screen and timing
SCREEN_W, SCREEN_H = 400, 600
FPS = 60
GROUND_H = 96
GROUND_Y = SCREEN_H - GROUND_H

# Gameplay constants
BASE_GRAVITY = 0.38
BASE_FLAP_STRENGTH = -7.8
BASE_PIPE_SPEED = 2.8
BASE_PIPE_GAP = 165
BASE_PIPE_INTERVAL = 1450  # ms between new pipes

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 215, 32)
ORANGE = (255, 140, 0)
SKY_TOP = (91, 192, 235)
SKY_BOTTOM = (130, 220, 255)
GROUND_C = (223, 216, 133)
GRASS = (146, 189, 67)
PIPE_GREEN = (102, 191, 64)
PIPE_DARK = (78, 152, 49)
TEXT_DARK = (24, 30, 52)
PANEL_BG = (36, 41, 77)
RED = (240, 63, 63)


@dataclass(frozen=True)
class Difficulty:
    name: str
    gravity_mult: float
    flap_mult: float
    speed_mult: float
    gap_offset: int
    pipe_interval: int


DIFFICULTIES = [
    Difficulty("Easy", 0.94, 1.04, 0.90, 18, 1600),
    Difficulty("Normal", 1.00, 1.00, 1.00, 0, 1450),
    Difficulty("Hard", 1.08, 0.97, 1.12, -20, 1300),
]


class Bird:
    W, H = 34, 24

    def __init__(self) -> None:
        self.x = 88
        self.reset()

    def reset(self) -> None:
        self.y = SCREEN_H // 2 - 40
        self.vel = 0.0
        self.rotation = 0.0

    def flap(self, flap_strength: float) -> None:
        self.vel = flap_strength

    def update(self, gravity: float) -> None:
        self.vel = min(self.vel + gravity, 10)
        self.y += self.vel
        self.rotation = max(-28, min(-self.vel * 4.1, 80))

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.x - self.W // 2 + 5,
            int(self.y) - self.H // 2 + 4,
            self.W - 10,
            self.H - 8,
        )

    def draw(self, surf: pygame.Surface) -> None:
        # Draw to a temporary surface, rotate, and blit so it feels closer to the original.
        tmp = pygame.Surface((60, 60), pygame.SRCALPHA)
        cx, cy = 30, 30

        body_rect = pygame.Rect(cx - self.W // 2, cy - self.H // 2, self.W, self.H)
        pygame.draw.ellipse(tmp, YELLOW, body_rect)
        pygame.draw.ellipse(tmp, ORANGE, body_rect, 2)

        wing_offset = 6 if self.vel < 0 else 10
        pygame.draw.ellipse(tmp, ORANGE, (cx - 12, cy + wing_offset - 6, 18, 10))
        pygame.draw.ellipse(tmp, (255, 180, 50), (cx - 8, cy + wing_offset - 4, 10, 6))

        pygame.draw.circle(tmp, WHITE, (cx + 7, cy - 4), 6)
        pygame.draw.circle(tmp, BLACK, (cx + 9, cy - 4), 3)

        beak = [(cx + 15, cy), (cx + 24, cy - 2), (cx + 15, cy + 4)]
        pygame.draw.polygon(tmp, ORANGE, beak)

        rotated = pygame.transform.rotozoom(tmp, self.rotation, 1.0)
        rect = rotated.get_rect(center=(self.x, int(self.y)))
        surf.blit(rotated, rect)


class Pipe:
    W = 64
    CAP_H = 24

    def __init__(self, gap: int):
        margin_top = 90
        margin_bottom = 110
        gap_y = random.randint(margin_top, SCREEN_H - margin_bottom - gap)
        self.top = pygame.Rect(SCREEN_W + 10, 0, self.W, gap_y)
        self.bottom = pygame.Rect(SCREEN_W + 10, gap_y + gap, self.W, SCREEN_H)
        self.scored = False

    def update(self, speed: float) -> None:
        dx = int(round(speed))
        self.top.x -= dx
        self.bottom.x -= dx

    def off_screen(self) -> bool:
        return self.top.right < -10

    def collides(self, bird_rect: pygame.Rect) -> bool:
        return bird_rect.colliderect(self.top) or bird_rect.colliderect(self.bottom)

    def draw(self, surf: pygame.Surface) -> None:
        for rect, top_cap in ((self.top, False), (self.bottom, True)):
            pygame.draw.rect(surf, PIPE_GREEN, rect)
            pygame.draw.rect(surf, PIPE_DARK, rect, 4)

            cap_y = rect.y if top_cap else rect.bottom - self.CAP_H
            cap = pygame.Rect(rect.x - 5, cap_y, self.W + 10, self.CAP_H)
            pygame.draw.rect(surf, PIPE_GREEN, cap)
            pygame.draw.rect(surf, PIPE_DARK, cap, 4)

            # simple vertical highlight
            hi = pygame.Rect(rect.x + 6, rect.y + 4, 6, max(6, rect.h - 8))
            pygame.draw.rect(surf, (146, 226, 115), hi)


class Cloud:
    def __init__(self, x: float | None = None):
        self.x = x if x is not None else SCREEN_W + random.randint(0, 180)
        self.y = random.randint(45, 200)
        self.speed = random.uniform(0.25, 0.75)
        self.size = random.randint(26, 54)

    def update(self) -> None:
        self.x -= self.speed

    def draw(self, surf: pygame.Surface) -> None:
        s = self.size
        for dx, dy, r in ((0, 0, s // 2), (s // 2, -s // 4, s // 3), (-s // 2, -s // 5, s // 3), (s, 0, s // 3)):
            pygame.draw.circle(surf, WHITE, (int(self.x + dx), int(self.y + dy)), r)


def draw_button(
    surf: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    *,
    active: bool = False,
    pressed: bool = False,
) -> None:
    fill = (255, 222, 93) if active else (233, 241, 255)
    border = (172, 126, 24) if active else (87, 102, 152)
    if pressed:
        fill = (220, 205, 137) if active else (208, 220, 240)
    pygame.draw.rect(surf, fill, rect, border_radius=12)
    pygame.draw.rect(surf, border, rect, 3, border_radius=12)

    label = font.render(text, True, TEXT_DARK)
    surf.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))


def draw_sky_gradient(surf: pygame.Surface) -> None:
    for y in range(SCREEN_H):
        t = y / SCREEN_H
        r = int(SKY_TOP[0] * (1 - t) + SKY_BOTTOM[0] * t)
        g = int(SKY_TOP[1] * (1 - t) + SKY_BOTTOM[1] * t)
        b = int(SKY_TOP[2] * (1 - t) + SKY_BOTTOM[2] * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (SCREEN_W, y))


def draw_ground(surf: pygame.Surface, offset: int) -> None:
    pygame.draw.rect(surf, GROUND_C, (0, GROUND_Y, SCREEN_W, GROUND_H))
    pygame.draw.rect(surf, GRASS, (0, GROUND_Y, SCREEN_W, 18))
    pygame.draw.line(surf, (112, 148, 44), (0, GROUND_Y + 17), (SCREEN_W, GROUND_Y + 17), 3)

    stripe_w = 36
    for i in range(-1, SCREEN_W // stripe_w + 3):
        x = i * stripe_w - (offset % stripe_w)
        pygame.draw.line(surf, (192, 184, 116), (x, GROUND_Y + 52), (x + 18, GROUND_Y + 52), 3)


def draw_centered_text(surf, text, font, color, y, shadow=True):
    if shadow:
        s = font.render(text, True, (25, 25, 25))
        surf.blit(s, (SCREEN_W // 2 - s.get_width() // 2 + 2, y + 2))
    t = font.render(text, True, color)
    surf.blit(t, (SCREEN_W // 2 - t.get_width() // 2, y))


def make_panel(size, alpha=220):
    panel = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*PANEL_BG, alpha), panel.get_rect(), border_radius=16)
    pygame.draw.rect(panel, (235, 235, 245, alpha), panel.get_rect(), 3, border_radius=16)
    return panel


def menu_button_rects() -> tuple[list[pygame.Rect], pygame.Rect]:
    diff_rects = [pygame.Rect(88, 262 + i * 56, 224, 44) for i in range(len(DIFFICULTIES))]
    start_rect = pygame.Rect(102, 442, 196, 52)
    return diff_rects, start_rect


def dead_button_rects() -> tuple[list[pygame.Rect], pygame.Rect]:
    diff_rects = [pygame.Rect(98, 336 + i * 44, 204, 36) for i in range(len(DIFFICULTIES))]
    retry_rect = pygame.Rect(108, 470, 184, 48)
    return diff_rects, retry_rect


def compute_viewport(window_size: tuple[int, int]) -> tuple[float, int, int, int, int]:
    win_w, win_h = window_size
    scale = min(win_w / SCREEN_W, win_h / SCREEN_H)
    view_w = max(1, int(SCREEN_W * scale))
    view_h = max(1, int(SCREEN_H * scale))
    off_x = (win_w - view_w) // 2
    off_y = (win_h - view_h) // 2
    return scale, off_x, off_y, view_w, view_h


def map_pointer_to_virtual(pos: tuple[int, int], viewport: tuple[float, int, int, int, int]) -> tuple[int, int] | None:
    scale, off_x, off_y, view_w, view_h = viewport
    px, py = pos
    if not (off_x <= px < off_x + view_w and off_y <= py < off_y + view_h):
        return None
    vx = int((px - off_x) / scale)
    vy = int((py - off_y) / scale)
    return max(0, min(SCREEN_W - 1, vx)), max(0, min(SCREEN_H - 1, vy))


def main() -> None:
    pygame.init()
    window = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    game_surface = pygame.Surface((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Flappy Bird")
    clock = pygame.time.Clock()

    font_huge = pygame.font.SysFont("Arial", 56, bold=True)
    font_big = pygame.font.SysFont("Arial", 42, bold=True)
    font_med = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 21, bold=False)
    font_tiny = pygame.font.SysFont("Arial", 16, bold=True)

    bird = Bird()
    pipes: list[Pipe] = []
    clouds = [Cloud(random.randint(0, SCREEN_W)) for _ in range(6)]
    state = "menu"  # menu | play | dead
    difficulty_idx = 1

    score = 0
    best = 0
    ground_offset = 0
    last_pipe = pygame.time.get_ticks()

    def active_diff() -> Difficulty:
        return DIFFICULTIES[difficulty_idx]

    def gameplay_values() -> tuple[float, float, float, int, int]:
        d = active_diff()
        return (
            BASE_GRAVITY * d.gravity_mult,
            BASE_FLAP_STRENGTH * d.flap_mult,
            BASE_PIPE_SPEED * d.speed_mult,
            BASE_PIPE_GAP + d.gap_offset,
            d.pipe_interval,
        )

    def start_game() -> None:
        nonlocal score, pipes, last_pipe, ground_offset, state
        score = 0
        bird.reset()
        pipes = []
        ground_offset = 0
        last_pipe = pygame.time.get_ticks() - active_diff().pipe_interval
        state = "play"
        _, flap, _, _, _ = gameplay_values()
        bird.flap(flap)

    running = True
    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()
        gravity, flap_strength, pipe_speed, pipe_gap, pipe_interval = gameplay_values()
        viewport = compute_viewport(window.get_size())

        menu_diffs, menu_start = menu_button_rects()
        dead_diffs, dead_retry = dead_button_rects()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            pointer_down = False
            physical_pos: tuple[int, int] | None = None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pointer_down = True
                physical_pos = event.pos
            elif event.type == pygame.FINGERDOWN:
                pointer_down = True
                win_w, win_h = window.get_size()
                physical_pos = (int(event.x * win_w), int(event.y * win_h))

            if pointer_down and physical_pos is not None:
                pointer_pos = map_pointer_to_virtual(physical_pos, viewport)
                if pointer_pos is None:
                    continue

                if state == "menu":
                    clicked_diff = False
                    for i, rect in enumerate(menu_diffs):
                        if rect.collidepoint(pointer_pos):
                            difficulty_idx = i
                            clicked_diff = True
                            break
                    if not clicked_diff and menu_start.collidepoint(pointer_pos):
                        start_game()

                elif state == "play":
                    bird.flap(flap_strength)

                elif state == "dead":
                    clicked_diff = False
                    for i, rect in enumerate(dead_diffs):
                        if rect.collidepoint(pointer_pos):
                            difficulty_idx = i
                            clicked_diff = True
                            break
                    if not clicked_diff and dead_retry.collidepoint(pointer_pos):
                        start_game()

        if state == "play":
            bird.update(gravity)
            ground_offset += int(round(pipe_speed))

            if now - last_pipe > pipe_interval:
                pipes.append(Pipe(pipe_gap))
                last_pipe = now

            bird_rect = bird.get_rect()
            for pipe in pipes:
                pipe.update(pipe_speed)
                if not pipe.scored and pipe.top.right < bird.x:
                    pipe.scored = True
                    score += 1
                    best = max(best, score)
                if pipe.collides(bird_rect):
                    state = "dead"

            pipes = [p for p in pipes if not p.off_screen()]

            if bird.y - bird.H // 2 <= 0:
                bird.y = bird.H // 2
                bird.vel = 0
            if bird.y + bird.H // 2 >= GROUND_Y:
                bird.y = GROUND_Y - bird.H // 2
                state = "dead"

        for c in clouds:
            c.update()
        clouds = [c for c in clouds if c.x > -120]
        if len(clouds) < 7 and random.random() < 0.015:
            clouds.append(Cloud())

        draw_sky_gradient(game_surface)
        for c in clouds:
            c.draw(game_surface)
        for pipe in pipes:
            pipe.draw(game_surface)

        draw_ground(game_surface, ground_offset)
        bird.draw(game_surface)

        if state in ("play", "dead"):
            draw_centered_text(game_surface, str(score), font_huge, WHITE, 34)

        if state == "menu":
            draw_centered_text(game_surface, "FLAPPY BIRD", font_big, YELLOW, 102)
            panel = make_panel((SCREEN_W - 58, 360), alpha=208)
            game_surface.blit(panel, (29, 168))
            draw_centered_text(game_surface, "Select Difficulty", font_med, WHITE, 205)

            for i, diff in enumerate(DIFFICULTIES):
                draw_button(game_surface, menu_diffs[i], diff.name, font_small, active=(i == difficulty_idx))

            draw_button(game_surface, menu_start, "START", font_med, active=True)

        if state == "dead":
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 85))
            game_surface.blit(overlay, (0, 0))

            panel = make_panel((SCREEN_W - 70, 420), alpha=228)
            game_surface.blit(panel, (35, 128))
            draw_centered_text(game_surface, "GAME OVER", font_big, RED, 158)
            draw_centered_text(game_surface, f"Score : {score}", font_med, WHITE, 236)
            draw_centered_text(game_surface, f"Best  : {best}", font_med, YELLOW, 274)
            draw_centered_text(game_surface, "Difficulty", font_small, (180, 225, 255), 306)
            for i, diff in enumerate(DIFFICULTIES):
                draw_button(game_surface, dead_diffs[i], diff.name, font_tiny, active=(i == difficulty_idx))
            draw_button(game_surface, dead_retry, "RETRY", font_small, active=True)

        _, off_x, off_y, view_w, view_h = viewport
        window.fill((0, 0, 0))
        scaled = pygame.transform.smoothscale(game_surface, (view_w, view_h))
        window.blit(scaled, (off_x, off_y))
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()