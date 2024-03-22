# PINNs-Training
Project where we study PINN methodology for solving various differential equations.

---

Because of increasing interest in deep learning methods for incorporating prior physical knowledge in data-driven methods, and young and popular method "PINNs" for doing so using governing differential equations, there is a lot of work about, for example, how to solve particular problem in particular domain, or how to increase our confidence about solution accuracy without explicitly comparing it to given ground truth (which is usually does not known), or how to create stable, fast and reliable PDE solver using PINN, etc. Much work has been done about how to implement PINNs in a specific circumstances or to represent specific property (simmetries, discontinuities, shocks and much more), so we are very rich from this point of view and understand this methodology well.

But we are facing a huge and somewhat unsolved problem - how to choose hyperparameters when using PINNs? This is simultaneously a theoretical question - because it raises a second-order questions, such as "why these hyperparameters are optimal", "does there exist connection between differential equation & problem statement and optimal hyperparameters" etc., - and a practical question - because we are usually want to solve a problem in efficient way and dont want to search them manually (we can always rely on automatic HPO methods and frameworks, but we then lack theoretical understanding on why and how such search should be performed).

There are plenty of basic conclusions like "more complex solution leads to bigger neural network" etc., but we are interested in something beyond that. So, our main questions is: a) how does hyperparameters depend on problem statement? and b) how should we choose hyperparameters? For clarity, we are working with basic PINN method, without much clever enhancements or tricky tricks with data and loss function. 

In this study we are focusing not only on how to solve particular differential equations, but we are sistematically investigate the problem described above. To-dos in the following section shows our finished work and our plans.

## Our Work

Differential equations we are dealing with are listed below:

- Damped Harmonic Oscillator (DHO): simple 1D ODE system of oscillating mass.
- Lotka-Volterra System (LV): 2D dynamical system of two dependent quantities.
- Lorenz System (LZ): famous 3D system with chaotic behavior.
- Advection Equation (Adv): 1D first-order hyperbolic PDE of quantity transport.
- Diffusion Equation (Diff): 1D second-order parabolic PDE of quantity diffusion.
- Wave Equation (Wave): 1D second-order hyperbolic PDE of wave travelling.
- Ornstein-Uhlenbeck Process (OU): simple stochastic differential equation in 1D.
- Burgers Equation (Burg): 1D nonlinear PDE of wave travelling that exhibits shocks.
- Korteweg-de Vries Equation (KdV): 1D nonlinear third-order PDE of wave travelling with high nonlinearity and soliton solutions.
- Schrodinger Equation (Schr): 1D second-order PDE of wave function evolution through time and complex numbers.
- Newton System (Newt): 2D second-order system of 2N ODE, where N is a number of particles.
- Primitive Equations (Prim): system of PDEs that describes evolution of the atmosphere.

Here are the tasks that we choose for our analysis:

1. Build a numerical method to obtain ground truth on demand.
2. Build PINN, solve one particular problem using it to check if it works good.
3. Choose 3-4 different problem statements with different and perhaps interesting behavior.
4. Solve them and obtain optimal hyperparameters.
5. Analyze, what makes difference between optimal hyperparameters, and how they are connected with problem statement and solution.
6. Get one interesting problem statement. Investigate, how does hyperparameters and training process changing with slight changes in problem statement (for example, perturb coefficients, final time T, boundary conditions; this analysis should provide information about smooth dependence of hyperparameters on problem statement).
7. Analysize different activation functions. Does there exist one good activation function that we should use, or they depend on particular problem?
8. Analyze different rules of initialization. There are good and bad ones; what exactly makes optimization process stable and fast? What good initialization means - does it have connections with a properties of solution? Why bad initial state leads to approximation of zero function?

We will update this table according to our progress:

| Name of System \ Task | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Damped Harmonic Oscillator | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lotka-Volterra System | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lorenz System | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Advection Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Diffusion Equation | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wave Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ornstein-Uhlenbeck Process | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Burgers Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Korteweg-de Vries Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Schrodinger Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Newtons Equations System | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Primitive Equations | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Hypotheses to check:

1. If neural network size increases, then required amount of collocation points does so.
   - This happens because of vanishing gradient problem.
   - Number of layers L affects this more than network width W.
2. If final time T increases, then required neural network size does so.
   - This happens because of increasing complexity of solution.
   - This happens because of increasing sparsity in collocation points (for example, 1000 points placed in equal distance on [0, 10] are much more densely located than 1000 points on [0, 100]) and maybe some consequences.
3. As a consequence of 1, not big enough batch of collocation points leads to approximation of zero function.
   - This happens because optimizer does not able to choose good direction of minimization and falls down to bad minima.
   - This phenomena corresponds to bad initialization in the following sense: not big enough batch leads to small "mistakes" in the direction of optimization, and eventually sticking in bad region of loss landscape, while bad initialization causes optimizer to make a huge "mistake". This "mistakes", however, happens in earlier stages of optimization, and both leads to approximation of zero function.
4. If a batch size becomes smaller, then learning rate should be smaller too.
   - This happens because of increasing noise in optimizer. Choosing smaller lr will prevent optimizer of falling to bad minima.
5. Phases of learning corresponds to learning of initial conditions and regularization rule (differential equation) at different times.

## Our Roles

Daniil Laptev: tasks 1-3, 5-8.

Daniil Faryshev: task 4.
