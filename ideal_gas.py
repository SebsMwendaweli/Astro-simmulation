"""
N-Particle Ideal Gas in a Box
Visualization: pygame
Collision model: elastic wall reflections only (no particle-particle collisions yet)
Box size: controllable at runtime

Controls:
  UP/DOWN     - Increase / decrease N (restart required — press R)
  LEFT/RIGHT  - Shrink / expand box width & height
  +/-         - Speed up / slow down particles (rescale velocities)
  R           - Reset with current N
  SPACE       - Pause/Resume
  ESC         - Quit

Physics notes:
  - Particles move in straight lines and bounce off walls elastically.
  - No particle-particle collisions (placeholder logic ready to add).
  - Speed distribution initialised as Maxwell-Boltzmann (2D).
"""

import pygame
import numpy as np
import sys

# ── Simulation defaults ────────────────────────────────────────────────────────
DEFAULT_N   = 80
RADIUS      = 6           # particle radius, pixels
MASS        = 1.0
TEMP_SCALE  = 200.0       # mean speed in pixels/s
DT          = 1 / 60      # seconds per frame

# Box defaults (pixels)
BOX_MIN_SIZE = 150
BOX_DEFAULT_W = 600
BOX_DEFAULT_H = 500

WIDTH, HEIGHT = 1000, 720
FPS = 60

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = (10, 10, 18)
BOX_COL   = (60, 60, 100)
BOX_FILL  = (15, 15, 28)
WHITE     = (240, 240, 255)
GREY      = (140, 140, 160)
FAST_COL  = (255, 80,  60)
SLOW_COL  = (60, 140, 255)
MID_COL   = (80, 220, 160)
HUD_BG    = (20, 20, 35, 180)


# ── Particle colour by speed ────────────────────────────────────────────────────
def speed_colour(speed, v_mean):
    t = np.clip(speed / (2 * v_mean + 1e-6), 0, 1)
    r = int(SLOW_COL[0] + t * (FAST_COL[0] - SLOW_COL[0]))
    g = int(SLOW_COL[1] + t * (FAST_COL[1] - SLOW_COL[1]))
    b = int(SLOW_COL[2] + t * (FAST_COL[2] - SLOW_COL[2]))
    return (r, g, b)


# ── Initialise particles ───────────────────────────────────────────────────────
def init_particles(n, box_x, box_y, box_w, box_h, temp_scale):
    """
    Place n particles randomly inside the box and give them
    2D Maxwell-Boltzmann speed distribution (Rayleigh).
    """
    rng = np.random.default_rng()

    # Positions: uniform inside box with margin = RADIUS
    margin = RADIUS + 1
    px = rng.uniform(box_x + margin, box_x + box_w - margin, n)
    py = rng.uniform(box_y + margin, box_y + box_h - margin, n)
    pos = np.column_stack([px, py])

    # Velocities: 2D Maxwell-Boltzmann
    sigma = temp_scale / np.sqrt(2)
    vel   = rng.normal(0, sigma, (n, 2))

    return pos.copy(), vel.copy()


# ── Wall collision ──────────────────────────────────────────────────────────────
def wall_collide(pos, vel, box_x, box_y, box_w, box_h):
    """Reflect particles off walls; returns updated pos and vel."""
    left   = box_x + RADIUS
    right  = box_x + box_w - RADIUS
    top    = box_y + RADIUS
    bottom = box_y + box_h - RADIUS

    # Left/right walls
    hit_left  = pos[:, 0] < left
    hit_right = pos[:, 0] > right
    pos[hit_left,  0] = left
    pos[hit_right, 0] = right
    vel[hit_left  | hit_right, 0] *= -1

    # Top/bottom walls
    hit_top    = pos[:, 1] < top
    hit_bottom = pos[:, 1] > bottom
    pos[hit_top,    1] = top
    pos[hit_bottom, 1] = bottom
    vel[hit_top | hit_bottom, 1] *= -1

    return pos, vel


# ── Placeholder: particle-particle collision (future) ─────────────────────────
def particle_collide(pos, vel):
    """
    TODO: elastic particle-particle collisions.
    Detect pairs where |r_i - r_j| < 2*RADIUS and apply
    1D elastic collision along the line of centres.
    Currently a no-op.
    """
    return pos, vel


# ── Statistics ──────────────────────────────────────────────────────────────────
def kinetic_stats(vel):
    speeds    = np.linalg.norm(vel, axis=1)
    v_mean    = speeds.mean()
    v_rms     = np.sqrt((speeds**2).mean())
    ke_total  = 0.5 * MASS * (speeds**2).sum()
    return speeds, v_mean, v_rms, ke_total


# ── Speed histogram ─────────────────────────────────────────────────────────────
def draw_histogram(surface, speeds, v_mean, rect):
    x0, y0, w, h = rect
    pygame.draw.rect(surface, (20, 20, 35), (x0, y0, w, h))
    pygame.draw.rect(surface, BOX_COL, (x0, y0, w, h), 1)

    bins    = 20
    v_max   = max(speeds.max(), 2 * v_mean, 1)
    counts, edges = np.histogram(speeds, bins=bins, range=(0, v_max))
    bar_w   = w / bins
    max_c   = counts.max() if counts.max() > 0 else 1

    for i, c in enumerate(counts):
        bar_h = int(h * c / max_c)
        bx    = int(x0 + i * bar_w)
        by    = y0 + h - bar_h
        mid_v = (edges[i] + edges[i+1]) / 2
        col   = speed_colour(mid_v, v_mean)
        pygame.draw.rect(surface, col, (bx+1, by, int(bar_w)-1, bar_h))


# ── HUD ────────────────────────────────────────────────────────────────────────
def draw_hud(surface, font, n, v_mean, v_rms, ke, box_w, box_h, temp_scale, paused):
    lines = [
        f"N          = {n}",
        f"<v>        = {v_mean:.1f} px/s",
        f"v_rms      = {v_rms:.1f} px/s",
        f"KE_total   = {ke:.0f}",
        f"Box        = {box_w}×{box_h} px",
        f"Temp scale = {temp_scale:.0f}",
        "",
        "↑↓ N  ←→ box size",
        "+/- speed  R reset",
    ]
    y = 10
    for line in lines:
        s = font.render(line, True, GREY)
        surface.blit(s, (WIDTH - 220, y))
        y += 20
    if paused:
        ps = font.render("PAUSED", True, (255, 80, 80))
        surface.blit(ps, (WIDTH - 220, y + 10))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ideal Gas — N Particles in a Box")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 15)

    n         = DEFAULT_N
    box_w     = BOX_DEFAULT_W
    box_h     = BOX_DEFAULT_H
    temp_scale= TEMP_SCALE
    box_x     = (WIDTH  - box_w) // 2
    box_y     = (HEIGHT - box_h) // 2

    pos, vel  = init_particles(n, box_x, box_y, box_w, box_h, temp_scale)
    paused    = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    box_x = (WIDTH  - box_w) // 2
                    box_y = (HEIGHT - box_h) // 2
                    pos, vel = init_particles(n, box_x, box_y, box_w, box_h, temp_scale)
                elif event.key == pygame.K_UP:
                    n = min(n + 10, 500)
                    box_x = (WIDTH  - box_w) // 2
                    box_y = (HEIGHT - box_h) // 2
                    pos, vel = init_particles(n, box_x, box_y, box_w, box_h, temp_scale)
                elif event.key == pygame.K_DOWN:
                    n = max(n - 10, 5)
                    box_x = (WIDTH  - box_w) // 2
                    box_y = (HEIGHT - box_h) // 2
                    pos, vel = init_particles(n, box_x, box_y, box_w, box_h, temp_scale)
                elif event.key == pygame.K_RIGHT:
                    box_w = min(box_w + 20, WIDTH  - 40)
                    box_h = min(box_h + 20, HEIGHT - 40)
                    box_x = (WIDTH  - box_w) // 2
                    box_y = (HEIGHT - box_h) // 2
                    # Clamp particles inside new box
                    pos, vel = wall_collide(pos, vel, box_x, box_y, box_w, box_h)
                elif event.key == pygame.K_LEFT:
                    box_w = max(box_w - 20, BOX_MIN_SIZE)
                    box_h = max(box_h - 20, BOX_MIN_SIZE)
                    box_x = (WIDTH  - box_w) // 2
                    box_y = (HEIGHT - box_h) // 2
                    pos, vel = wall_collide(pos, vel, box_x, box_y, box_w, box_h)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    temp_scale = min(temp_scale * 1.2, 1000)
                    speeds = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
                    vel    = vel / speeds * (temp_scale / np.sqrt(2))
                elif event.key == pygame.K_MINUS:
                    temp_scale = max(temp_scale / 1.2, 20)
                    speeds = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
                    vel    = vel / speeds * (temp_scale / np.sqrt(2))

        if not paused:
            pos += vel * DT
            pos, vel = wall_collide(pos, vel, box_x, box_y, box_w, box_h)
            pos, vel = particle_collide(pos, vel)   # no-op for now

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Box
        pygame.draw.rect(screen, BOX_FILL, (box_x, box_y, box_w, box_h))
        pygame.draw.rect(screen, BOX_COL,  (box_x, box_y, box_w, box_h), 2)

        speeds, v_mean, v_rms, ke = kinetic_stats(vel)

        # Particles
        for i in range(n):
            col = speed_colour(speeds[i], v_mean)
            pygame.draw.circle(screen, col, (int(pos[i, 0]), int(pos[i, 1])), RADIUS)
            # Tiny highlight
            pygame.draw.circle(screen, WHITE,
                               (int(pos[i, 0]) - RADIUS//3, int(pos[i, 1]) - RADIUS//3),
                               max(1, RADIUS//4))

        # Histogram
        hist_rect = (10, HEIGHT - 180, 250, 160)
        draw_histogram(screen, speeds, v_mean, hist_rect)
        s = font.render("Speed distribution", True, GREY)
        screen.blit(s, (10, HEIGHT - 195))

        draw_hud(screen, font, n, v_mean, v_rms, ke, box_w, box_h, temp_scale, paused)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main
