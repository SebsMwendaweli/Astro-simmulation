"""
Two-Body Gravitational Simulation
Numerical method: Velocity Verlet
Visualization: pygame

Controls:
  R - Reset simulation
  SPACE - Pause/Resume
  ESC - Quit
"""

import pygame
import numpy as np
import sys

# ── Constants ─────────────────────────────────────────────────────────────────
G  = 6.674e-11   # gravitational constant (SI)
AU = 1.496e11    # 1 astronomical unit in metres
M_SUN = 1.989e30 # solar mass in kg

# Simulation scale
SCALE = 200 / AU        # pixels per metre  (200 px = 1 AU)
DT    = 3600 * 24       # time step: 1 day in seconds
WIDTH, HEIGHT = 900, 700
FPS   = 60

TRAIL_LEN = 800         # max trail points per body

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = (8, 8, 20)
STAR_COL  = (255, 220, 80)
PLANET_COL= (80, 160, 255)
TRAIL1    = (180, 140, 40)
TRAIL2    = (40, 100, 200)
WHITE     = (255, 255, 255)
GREY      = (160, 160, 160)


# ── Physics helpers ────────────────────────────────────────────────────────────
def gravitational_acc(pos1: np.ndarray, pos2: np.ndarray, mass2: float) -> np.ndarray:
    """Acceleration on body 1 due to body 2."""
    r_vec = pos2 - pos1
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e6:          # softening to avoid singularity
        r_mag = 1e6
    return G * mass2 / r_mag**3 * r_vec


def velocity_verlet_step(pos1, vel1, pos2, vel2, m1, m2, dt):
    """
    Full Velocity Verlet step for both bodies.

    x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    a'      = a( x(t+dt) )
    v(t+dt) = v(t) + 0.5*(a(t) + a')*dt
    """
    # Accelerations at current positions
    a1 = gravitational_acc(pos1, pos2, m2)
    a2 = gravitational_acc(pos2, pos1, m1)

    # Update positions
    new_pos1 = pos1 + vel1 * dt + 0.5 * a1 * dt**2
    new_pos2 = pos2 + vel2 * dt + 0.5 * a2 * dt**2

    # Accelerations at new positions
    a1_new = gravitational_acc(new_pos1, new_pos2, m2)
    a2_new = gravitational_acc(new_pos2, new_pos1, m1)

    # Update velocities
    new_vel1 = vel1 + 0.5 * (a1 + a1_new) * dt
    new_vel2 = vel2 + 0.5 * (a2 + a2_new) * dt

    return new_pos1, new_vel1, new_pos2, new_vel2


def to_screen(pos: np.ndarray, cx: float, cy: float) -> tuple:
    """Convert physical coords (metres) to screen pixels, centred on (cx,cy)."""
    return (int(cx + pos[0] * SCALE), int(cy - pos[1] * SCALE))


def total_energy(pos1, vel1, pos2, vel2, m1, m2):
    ke = 0.5 * m1 * np.dot(vel1, vel1) + 0.5 * m2 * np.dot(vel2, vel2)
    r  = np.linalg.norm(pos2 - pos1)
    pe = -G * m1 * m2 / r
    return ke + pe


# ── Initial conditions (Sun-Earth-like) ────────────────────────────────────────
def initial_state():
    m1 = M_SUN
    m2 = 0.3 * M_SUN        # equal-ish masses for interesting motion

    # Positions: symmetric about origin
    pos1 = np.array([-0.5 * AU, 0.0])
    pos2 = np.array([ 0.5 * AU, 0.0])

    # Circular orbit speed estimate: v = sqrt(G*M_other / (2r))
    r_half = 0.5 * AU
    v1 = np.sqrt(G * m2 / (2 * r_half)) * 0.95
    v2 = np.sqrt(G * m1 / (2 * r_half)) * 0.95

    vel1 = np.array([0.0,  v1])
    vel2 = np.array([0.0, -v2])

    return pos1, vel1, pos2, vel2, m1, m2


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_trail(surface, trail, colour):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha = int(255 * i / len(trail))
        c = tuple(int(ch * alpha / 255) for ch in colour)
        pygame.draw.line(surface, c, trail[i - 1], trail[i], 1)


def draw_info(surface, font, energy0, energy, t_days, paused):
    de = abs((energy - energy0) / energy0) * 100 if energy0 != 0 else 0
    lines = [
        f"Time: {t_days:.1f} days",
        f"ΔE/E₀: {de:.4f}%",
        "PAUSED" if paused else "",
        "[R] Reset  [SPACE] Pause  [ESC] Quit",
    ]
    y = 10
    for line in lines:
        col = (255, 80, 80) if line == "PAUSED" else GREY
        surf = font.render(line, True, col)
        surface.blit(surf, (10, y))
        y += 22


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Two-Body Gravitational Simulation — Velocity Verlet")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 16)

    cx, cy = WIDTH / 2, HEIGHT / 2

    pos1, vel1, pos2, vel2, m1, m2 = initial_state()
    energy0 = total_energy(pos1, vel1, pos2, vel2, m1, m2)

    trail1, trail2 = [], []
    t = 0.0
    paused = False
    steps_per_frame = 20   # sub-step for smoother integration

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    pos1, vel1, pos2, vel2, m1, m2 = initial_state()
                    energy0 = total_energy(pos1, vel1, pos2, vel2, m1, m2)
                    trail1.clear(); trail2.clear()
                    t = 0.0
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            for _ in range(steps_per_frame):
                pos1, vel1, pos2, vel2 = velocity_verlet_step(
                    pos1, vel1, pos2, vel2, m1, m2, DT
                )
                t += DT

            # Record trails
            trail1.append(to_screen(pos1, cx, cy))
            trail2.append(to_screen(pos2, cx, cy))
            if len(trail1) > TRAIL_LEN: trail1.pop(0)
            if len(trail2) > TRAIL_LEN: trail2.pop(0)

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Faint grid
        for x in range(0, WIDTH, 80):
            pygame.draw.line(screen, (20, 20, 35), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 80):
            pygame.draw.line(screen, (20, 20, 35), (0, y), (WIDTH, y))

        draw_trail(screen, trail1, TRAIL1)
        draw_trail(screen, trail2, TRAIL2)

        p1_screen = to_screen(pos1, cx, cy)
        p2_screen = to_screen(pos2, cx, cy)

        # Bodies
        pygame.draw.circle(screen, STAR_COL,   p1_screen, 14)
        pygame.draw.circle(screen, PLANET_COL, p2_screen, 10)
        # Glow rings
        pygame.draw.circle(screen, (255, 240, 150, 60), p1_screen, 20, 1)
        pygame.draw.circle(screen, (100, 180, 255, 60), p2_screen, 16, 1)

        energy = total_energy(pos1, vel1, pos2, vel2, m1, m2)
        draw_info(screen, font, energy0, energy, t / 86400, paused)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
