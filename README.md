# PINNs-Training
Project where we study PINN methodology for solving various differential equations.

---

Because of increasing interest in deep learning methods for incorporating prior physical knowledge in data-driven methods, and young and popular method "PINNs" for doing so using governing differential equations, there is a lot of work about, for example, how to solve particular problem in particular domain, or how to increase our confidence about solution accuracy without explicitly comparing it to given ground truth (which is usually does not known), or how to create stable, fast and reliable PDE solver using PINN, etc. Much work has been done about how to implement PINNs in a specific circumstances or to represent specific property (simmetries, discontinuities, shocks and much more), so we are very rich from this point of view and understand this methodology well.

But we are facing a huge and somewhat unsolved problem - how to choose hyperparameters when using PINNs? This is simultaneously a theoretical question - because it raises a second-order questions, such as "why these hyperparameters are optimal", "does there exist connection between differential equation & problem statement and optimal hyperparameters" etc., - and a practical question - because we are usually want to solve a problem in efficient way and dont want to search them manually (we can always rely on automatic HPO methods and frameworks, but we then lack theoretical understanding on why and how such search should be performed).

In this study we are focusing not only on how to solve particular differential equations, but we are sistematically investigate the problem described above. To-do lists in the following section shows our finished work and our plans.

## Finished and Future Work

Differential equations we are dealing with are listed below:

| Name of System | Differential Equation | Problem Statement |
|---|---|---|
| Damped Harmonic Oscillator (DHO) |  $\frac{\mathrm{d}^2 x}{\mathrm{d} t^2} + 2\zeta \omega_0\frac {\mathrm {d} x}{\mathrm {d} t} + \omega _0^2x = 0$ | $\begin{matrix} x(0) = x_0\\ \frac{\mathrm{d}x}{\mathrm{d}t}(0) = v_0 \end{matrix}$ |
| Lotka-Volterra System (LV) | $\begin{cases} \frac{\mathrm{d}x}{\mathrm{d}t} = \alpha x - \beta x y \\ \frac{\mathrm{d}y}{\mathrm{d}t} = \delta y x - \gamma y \end{cases}$ | $\begin{matrix} x(0) = x_0\\ y(0) = y_0 \end{matrix}$ |
| Diffusion Equation (Diffusion)  | $\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}$ | $\begin{matrix} u(x, 0) = f(x)\\ u(A, t) = g_1(t)\\ u(B, t) = g_2(t) \end{matrix}$ | 
| Lorenz System (LZ) | $\begin{cases} \frac{\mathrm{d}x}{\mathrm{d}t} = \sigma (y - x) \\ \frac{\mathrm{d}y}{\mathrm{d}t} = x(p - z) - y \\ \frac{\mathrm{d}z}{\mathrm{d}t} = xy - \beta z \end{cases}$ | $\begin{matrix} x(0) = x_0 \\ y(0) = y_0 \\ z(0) = z_0 \end{matrix}$ | 
| Wave Equation (Wave) | $\frac{\partial^2 u}{\partial t} = v \frac{\partial^2 u}{\partial x^2}$ | $\begin{matrix} u(x, 0) = f(x) \\ \frac{\partial u}{\partial t}(x, 0) = g(x) \end{matrix}$ | 
| Burgers Equation (Burgers) |  |  |
| Korteweg-de Vriez Equation (KdV) |  |  |  