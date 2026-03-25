"""
Simple Pendulum Simulation
Numerical method: 4th-order Runge-Kutta (RK4)
Visualization: pygame

State vector: [theta, omega]   (angle in radians, angular velocity in rad/s)
EOM: d²θ/dt² = -(g/L)*sin(θ)

Controls:
  R          - Reset to initial angle
  UP/DOWN    - Increase / decrease initial angle by 5°
  LEFT/RIGHT - Increase / decrease damping
  SPACE      - Pause/Resume
  ESC        - Quit
"""

import pygame
import numpy as np
import sys

# ── Simulation parameters ──────────────────────────────────────────────────────
G  = 9.81          # m/s²
L  = 2.0           # pendulum length, metres
DT = 0.01          # time step, seconds

WIDTH, HEIGHT = 900, 700
FPS = 60
STEPS_PER_FRAME = 3

PIVOT = (WIDTH // 2, 150)
PIX_PER_METRE = 130

TRAIL_LEN = 600

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = (12, 10, 18)
ROD_COL   = (180, 180, 200)
BOB_COL   = (80, 200, 180)
PIVOT_COL = (220, 220, 240)
TRAIL_COL = (80, 200, 180)
GREY      = (140, 140, 160)
WHITE     = (240, 240, 255)


# ── RK4 integrator ─────────────────────────────────────────────────────────────
def derivatives(state: np.ndarray, g: float, L: float, b: float) -> np.ndarray:
    """
    state = [theta, omega]
    returns [dtheta/dt, domega/dt]
    """
    theta, omega = state
    dtheta = omega
    domega = -(g / L) * np.sin(theta) - b * omega
    return np.array([dtheta, domega])


def rk4_step(state: np.ndarray, dt: float, g: float, L: float, b: float) -> np.ndarray:
    k1 = derivatives(state,            g, L, b)
    k2 = derivatives(state + 0.5*dt*k1, g, L, b)
    k3 = derivatives(state + 0.5*dt*k2, g, L, b)
    k4 = derivatives(state +     dt*k3, g, L, b)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ── Coordinate helpers ─────────────────────────────────────────────────────────
def bob_screen_pos(theta: float) -> tuple:
    x = PIVOT[0] + PIX_PER_METRE * L * np.sin(theta)
    y = PIVOT[1] + PIX_PER_METRE * L * np.cos(theta)
    return (int(x), int(y))


# ── Energy ─────────────────────────────────────────────────────────────────────
def energy(state, g, L, m=1.0):
    theta, omega = state
    ke = 0.5 * m * (L * omega)**2
    pe = m * g * L * (1 - np.cos(theta))
    return ke, pe, ke + pe


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_trail(surface, trail):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        t_frac = i / len(trail)
        alpha  = int(220 * t_frac)
        c = (int(TRAIL_COL[0]*t_frac), int(TRAIL_COL[1]*t_frac), int(TRAIL_COL[2]*t_frac))
        pygame.draw.line(surface, c, trail[i-1], trail[i], 1)


def draw_angle_arc(surface, theta):
    """Draw a small arc showing the current angle."""
    radius = 60
    start_angle = np.pi / 2           # pointing down (pygame y-axis)
    end_angle   = np.pi / 2 - theta
    if abs(theta) > 0.01:
        rect = pygame.Rect(PIVOT[0]-radius, PIVOT[1]-radius, 2*radius, 2*radius)
        a0 = min(start_angle, end_angle)
        a1 = max(start_angle, end_angle)
        pygame.draw.arc(surface, (200, 200, 100), rect, a0, a1, 1)


def draw_hud(surface, font, state, b, t, theta0_deg, paused, energy_vals):
    theta_deg = np.degrees(state[0])
    omega     = state[1]
    ke, pe, e = energy_vals
    lines = [
        f"θ  = {theta_deg:+.2f}°",
        f"ω  = {omega:+.3f} rad/s",
        f"KE = {ke:.4f} J/kg",
        f"PE = {pe:.4f} J/kg",
        f"E  = {e:.4f} J/kg",
        f"Damping b = {b:.3f}",
        f"L = {L:.1f} m  |  θ₀ = {theta0_deg:.0f}°",
        f"Time = {t:.2f} s",
        "",
        "↑↓ angle  ←→ damping",
        "[R] Reset  [SPACE] Pause",
    ]
    y = 10
    for line in lines:
        col = (255, 80, 80) if paused and line == "" else GREY
        surf = font.render(line, True, GREY)
        surface.blit(surf, (10, y))
        y += 20
    if paused:
        ps = font.render("PAUSED", True, (255, 80, 80))
        surface.blit(ps, (WIDTH - 100, 10))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simple Pendulum — RK4")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 16)

    theta0_deg = 60.0
    damping    = 0.0
    state      = np.array([np.radians(theta0_deg), 0.0])
    trail      = []
    t          = 0.0
    paused     = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = np.array([np.radians(theta0_deg), 0.0])
                    trail.clear(); t = 0.0
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    theta0_deg = min(theta0_deg + 5, 175)
                    state = np.array([np.radians(theta0_deg), 0.0])
                    trail.clear(); t = 0.0
                elif event.key == pygame.K_DOWN:
                    theta0_deg = max(theta0_deg - 5, 5)
                    state = np.array([np.radians(theta0_deg), 0.0])
                    trail.clear(); t = 0.0
                elif event.key == pygame.K_LEFT:
                    damping = max(0.0, damping - 0.05)
                elif event.key == pygame.K_RIGHT:
                    damping = min(damping + 0.05, 2.0)

        if not paused:
            for _ in range(STEPS_PER_FRAME):
                state = rk4_step(state, DT, G, L, damping)
                t += DT

            bob = bob_screen_pos(state[0])
            trail.append(bob)
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Background circles (atmosphere)
        for r in range(80, 300, 60):
            pygame.draw.circle(screen, (20, 18, 30), PIVOT, r, 1)

        draw_angle_arc(screen, state[0])
        draw_trail(screen, trail)

        bob = bob_screen_pos(state[0])

        # Vertical reference line
        pygame.draw.line(screen, (40, 40, 60), PIVOT,
                         (PIVOT[0], PIVOT[1] + int(PIX_PER_METRE * L) + 20), 1)

        # Rod
        pygame.draw.line(screen, ROD_COL, PIVOT, bob, 3)

        # Pivot
        pygame.draw.circle(screen, PIVOT_COL, PIVOT, 8)
        pygame.draw.circle(screen, BG, PIVOT, 4)

        # Bob with glow
        pygame.draw.circle(screen, (20, 80, 70), bob, 22)
        pygame.draw.circle(screen, BOB_COL, bob, 16)
        pygame.draw.circle(screen, WHITE, bob, 5)

        ev = energy(state, G, L)
        draw_hud(screen, font, state, damping, t, theta0_deg, paused, ev)

        # Energy bar
        bar_x, bar_y, bar_w, bar_h = WIDTH - 30, 50, 20, 300
        e_total = ev[2]
        e_max   = G * L * (1 - np.cos(np.radians(theta0_deg))) + 1e-9
        ke_frac = min(ev[0] / e_max, 1.0)
        pe_frac = min(ev[1] / e_max, 1.0)
        pygame.draw.rect(screen, (30, 30, 50), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(screen, (80, 200, 180),
                         (bar_x, bar_y + bar_h - int(ke_frac * bar_h), bar_w, int(ke_frac * bar_h)))
        pygame.draw.rect(screen, (200, 150, 80),
                         (bar_x, bar_y + bar_h - int((ke_frac+pe_frac)*bar_h/2),
                          bar_w, 2))
        for label, yy in [("E", bar_y-15), ("0", bar_y+bar_h)]:
            s = font.render(label, True, GREY)
            screen.blit(s, (bar_x, yy))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
