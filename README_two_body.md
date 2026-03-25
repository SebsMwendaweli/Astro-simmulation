# Two-Body Gravitational Simulation

A 2D simulation of the gravitational two-body problem using the **Velocity Verlet** integration scheme, visualised with **pygame**.

---

## Physics

The simulation solves Newton's law of gravitation for two point masses:

$$\ddot{\mathbf{r}}_1 = \frac{G m_2}{|\mathbf{r}_2 - \mathbf{r}_1|^3}(\mathbf{r}_2 - \mathbf{r}_1)$$

and symmetrically for body 2.

### Velocity Verlet Algorithm

At each time step:

1. Compute accelerations at current positions: `a(t)`
2. Update positions: `x(t+dt) = x(t) + v(t)·dt + ½·a(t)·dt²`
3. Compute accelerations at new positions: `a(t+dt)`
4. Update velocities: `v(t+dt) = v(t) + ½·[a(t) + a(t+dt)]·dt`

This is a symplectic integrator — it conserves energy over long times far better than simple Euler or RK4 for Hamiltonian systems.

---

## Default Initial Conditions

| Quantity | Value |
|---|---|
| Mass 1 | 1 M☉ |
| Mass 2 | 0.3 M☉ |
| Separation | 1 AU |
| Time step | 1 day |
| Sub-steps/frame | 20 |

Bodies are initialised with velocities approximating a circular orbit. The asymmetry in masses produces interesting precessing elliptical motion.

---

## Features

- Live energy conservation monitor `ΔE/E₀` — a direct diagnostic of integrator accuracy
- Fading colour trails for each body
- Softening length applied near `r → 0` to avoid singularity
- Coordinate system centred on screen; physical scale: 200 px = 1 AU

---

## Controls

| Key | Action |
|---|---|
| `R` | Reset simulation to initial conditions |
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
python two_body.py
```

---

## Notes & Extensions

- To change initial masses or orbital parameters, edit the `initial_state()` function.
- To experiment with different time steps, adjust `DT` (currently `3600 * 24` seconds = 1 day). Smaller `DT` improves accuracy at the cost of speed.
- Sub-steps per frame (`steps_per_frame`) can be increased for faster simulation without affecting visual smoothness.
- Suggested extension: add a centre-of-mass frame transform, or generalise to three bodies.
