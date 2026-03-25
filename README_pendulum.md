# Simple Pendulum Simulation

A simulation of the nonlinear simple pendulum using the **4th-order Runge-Kutta (RK4)** integrator, visualised with **pygame**.

---

## Physics

The equation of motion is the full nonlinear ODE — no small-angle approximation:

$$\ddot{\theta} = -\frac{g}{L}\sin\theta - b\dot{\theta}$$

where:
- `θ` — angle from vertical (radians)
- `g` — gravitational acceleration (9.81 m/s²)
- `L` — pendulum length (2.0 m)
- `b` — damping coefficient (adjustable, default 0)

The state vector is `[θ, ω]` where `ω = dθ/dt`.

### RK4 Algorithm

At each time step, four slope estimates are computed:

```
k1 = f(y)
k2 = f(y + dt/2 · k1)
k3 = f(y + dt/2 · k2)
k4 = f(y + dt  · k3)

y(t+dt) = y(t) + (dt/6) · (k1 + 2k2 + 2k3 + k4)
```

RK4 has local truncation error O(dt⁵) and global error O(dt⁴), making it well-suited for oscillatory systems.

---

## Default Parameters

| Parameter | Value |
|---|---|
| Pendulum length L | 2.0 m |
| Gravity g | 9.81 m/s² |
| Initial angle θ₀ | 60° |
| Damping b | 0.0 (undamped) |
| Time step dt | 0.01 s |
| Sub-steps/frame | 3 |

---

## Features

- Full nonlinear dynamics — valid for all amplitudes including near-180°
- Optional linear damping term, adjustable at runtime
- Live readout of `θ`, `ω`, kinetic energy, potential energy, and total energy
- Visual energy bar (right side of screen) showing KE in real time
- Small arc drawn at pivot to show current displacement angle
- Bob trail showing recent trajectory

---

## Controls

| Key | Action |
|---|---|
| `↑` / `↓` | Increase / decrease initial angle by 5° (resets simulation) |
| `←` / `→` | Decrease / increase damping coefficient |
| `R` | Reset to current initial angle with zero velocity |
| `SPACE` | Pause / Resume |
| `ESC` | Quit |

---

## Requirements

```
python >= 3.8
pygame
numpy
```

Install dependencies:

```bash
pip install pygame numpy
```

Run:

```bash
python pendulum_rk4.py
```

---

## Notes & Extensions

- At large amplitudes (e.g. 170°) you can observe how the period deviates significantly from the small-angle approximation `T = 2π√(L/g)`.
- With damping enabled, energy dissipation is visible in both the trail shortening and the energy bar.
- Suggested extensions:
  - Compare RK4 trajectory against the small-angle analytic solution.
  - Add a driven term `F·cos(ωt)` to the EOM to explore resonance and chaos (the driven damped pendulum).
  - Extend to a double pendulum by adding a second state vector.
