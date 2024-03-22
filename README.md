# PINNs-Training
Project where we study PINN methodology for solving various differential equations.

---

Because of increasing interest in deep learning methods for incorporating prior physical knowledge in data-driven methods, and young and popular method "PINNs" for doing so using governing differential equations, there is a lot of work about, for example, how to solve particular problem in particular domain, or how to increase our confidence about solution accuracy without explicitly comparing it to given ground truth (which is usually does not known), or how to create stable, fast and reliable PDE solver using PINN, etc. Much work has been done about how to implement PINNs in a specific circumstances or to represent specific property (simmetries, discontinuities, shocks and much more), so we are very rich from this point of view and understand this methodology well.

But we are facing a huge and somewhat unsolved problem - how to choose hyperparameters when using PINNs? This is simultaneously a theoretical question - because it raises a second-order questions, such as "why these hyperparameters are optimal", "does there exist connection between differential equation & problem statement and optimal hyperparameters" etc., - and a practical question - because we are usually want to solve a problem in efficient way and dont want to search them manually (we can always rely on automatic HPO methods and frameworks, but we then lack theoretical understanding on why and how such search should be performed).

In this study we are focusing not only on how to solve particular differential equations, but we are sistematically investigate the problem described above. To-do lists in the following section shows our finished work and our plans.

## Finished and Future Work

Differential equations we are dealing with are listed below:

- Damped Harmonic Oscillator (DHO): simple 1D ODE system.
- Lotka-Volterra System (LV): 2D dynamical system.
- Lorenz System (LZ): famous 3D system with chaotic behavior.
- Advection Equation (Adv): 1D first-order hyperbolic PDE.
- Diffusion Equation (Diff): 1D second-order parabolic PDE.
- Wave Equation (Wave): 1D second-order hyperbolic PDE.
- Burgers Equation (Burg): 1D nonlinear PDE that exhibits shocks.

| Name of System | Ground Truth Obtained | PINN Constructed | Optimal HPs  Found |
|---|:---:|:---:|:---:|
| Oscillator | ✅ | ✅ | ✅ |
| Lotka-Volterra | ✅ | ✅ | ✅ |
| Lorenz | ❌ | ❌ | ❌ |
| Advection | ❌ | ❌ | ❌ |
| Diffusion | ✅ | ✅ | ✅ |
| Wave | ❌ | ❌ | ❌ |
| Burgers | ❌ | ❌ | ❌ |
| Korteweg-de Vriez | ❌ | ❌ | ❌ |
| Schrodinger | ❌ | ❌ | ❌ |
| Newton | ❌ | ❌ | ❌ |
| Ornstein-Uhlenbeck | ❌ | ❌ | ❌ |
| Black-Scholes | ❌ | ❌ | ❌ |
| Primitive | ❌ | ❌ | ❌ |
|  |  |  |  |