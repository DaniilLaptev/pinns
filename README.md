# PINNs-Training
Project where we study PINN methodology for solving various differential equations.

---

Because of increasing interest in deep learning methods for incorporating prior physical knowledge in data-driven methods, and young and popular method "PINNs" for doing so using governing differential equations, there is a lot of work about, for example, how to solve particular problem in particular domain, or how to increase our confidence about solution accuracy without explicitly comparing it to given ground truth (which is usually does not known), or how to create stable, fast and reliable PDE solver using PINN, etc. Much work has been done about how to implement PINNs in a specific circumstances or to represent specific property (simmetries, discontinuities, shocks and much more), so we are very rich from this point of view and understand this methodology well.

But we are facing a huge and somewhat unsolved problem - how to choose hyperparameters when using PINNs? This is simultaneously a theoretical question - because it raises a second-order questions, such as "why these hyperparameters are optimal", "does there exist connection between differential equation & problem statement and optimal hyperparameters" etc., - and a practical question - because we are usually want to solve a problem in efficient way and dont want to search them manually (we can always rely on automatic HPO methods and frameworks, but we then lack theoretical understanding on why and how such search should be performed).

So, our main questions is: a) how does hyperparameters depend on problem statement? and b) how should we choose hyperparameters? There are plenty of basic conclusions like "more complex solution leads to bigger neural network" etc., but we are interested in something beyond that, for example, in smooth dependence of hyperparameters on problem statement or in hyperparameter "regions" for differential equations. For clarity, we are working with basic PINN method, without much clever enhancements or tricky tricks with data and loss function. 

In this study we are focusing not only on how to solve particular differential equations, but we are sistematically investigate the problem described above. To-dos in the following section shows our finished work and our plans.

## Our Work

Differential equations we are dealing with are listed below:

- Damped Harmonic Oscillator: simple 1D ODE system of oscillating mass.
- Lotka-Volterra System: 2D dynamical system of two dependent quantities.
- Lorenz System: famous 3D system with chaotic behavior.
- Diffusion Equation: 1D second-order parabolic PDE of quantity diffusion.
- Wave Equation: 1D second-order hyperbolic PDE of wave travelling.
- Laplace Equation: 2D second-order parabolic PDE that describes stationary or equilibrium state of a system.
- Gray-Scott Model: 2D system of PDE that describes reaction and diffusion process of two chemicals.
- Ornstein-Uhlenbeck Process: simple stochastic differential equation in 1D.
- Schrodinger Equation: 1D second-order PDE of wave function evolution through time and complex numbers.
- Newton System: 2D second-order system of 2N ODE, where N is a number of particles.
- Primitive Equations: system of PDEs that describes evolution of the atmosphere.
- Einstein Field Equations: system of PDEs that describes geometry of space-time and evolution of Universe.

Here are the tasks that we choose for our analysis:

1. Build a numerical method to obtain ground truth on demand.
2. Build PINN, solve one particular problem using it to check if it works good.
3. Choose 3-4 different problem statements with different and perhaps interesting behavior.
4. Choose optimal initialization rule for them. Solve problems and obtain optimal hyperparameters.
5. Analyze, what makes difference between optimal hyperparameters, and how they may be connected with a problem statement and solution.
6. Get two interesting problem statements and appropriate RMSE for each system. Investigate, how should we change hyperparameters to reach this RMSE if problem statement slightly changes (for example, perturb coefficients, final time T, boundary conditions; this analysis may provide information about smooth dependence of hyperparameters on problem statement). Look what happens with training process (why may we need more training iterations?).
7. Analysize different activation functions. Does there exist one good activation function that we should use, or they depend on particular problem?
8. Analyze different rules of initialization. There are good and bad ones; what exactly makes optimization process stable and fast? What good initialization means - does it have connections with a properties of solution? Why bad initial state leads to approximation of zero function?
9.  For some equations, analyze, what happens when we switch from one spatial dimension to two or three. Does conclusions, derived from tasks 4-8, remains valid? Any new phenomena occures?

We will update this table according to our progress:

| Name of System \ Task | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Damped Harmonic Oscillator | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lotka-Volterra System | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Lorenz System | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Diffusion Equation | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Gray-Scott Model | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wave Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ornstein-Uhlenbeck Process | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Schrodinger Equation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Newton System | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Primitive Equations | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Einstein Field Equations | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Hypotheses to check:

| # | Hypothesis or Best (Known) Explanation | Truth Value | Confirmed |
|:---:|:----|:---:|:---:|
| 1 | If neural network size increases, then required amount of collocation points does so. | True | No |
| a | This happens because of vanishing gradient problem. | True | No |
| b | Number of layers L affect this more than network width W. | True | No |
| 2 | If final time T increases, then required neural network size does so. | Not Always True | No |
| a | This happens because of increasing complexity of solution. | True | No |
| b | This happens because of increasing sparsity in collocation points (for example, 1000 points placed in equal distance on [0, 10] are much more densely located than 1000 points on [0, 100]) and maybe some consequences. | Unknown | No |
| c | This happens because during a learning process neural network learns solution function partially, property after property (and for Cauchy problem for ODE, from left to right). If T increases, properties becomes more complicated. | True | No |
| 3 | As a consequence of 1, not big enough batch of collocation points leads to approximation of zero function. | True | No |
| a | This happens because optimizer does not able to choose good direction of minimization and falls down to bad minima. | Unknown | No |
| b | This phenomena corresponds to bad initialization in the following sense: not big enough batch leads to small "mistakes" in the direction of optimization, and eventually sticking in bad region of loss landscape, while bad initialization causes optimizer to make a huge "mistake". This "mistakes", however, happens in earlier stages of optimization, and both leads to approximation of zero function. | True | No |
| 4 | If a batch size becomes smaller, then learning rate should be smaller too. | Unknown | No |
| a | This happens because of increasing noise in optimizer. Choosing smaller learning rate could prevent optimizer of falling to bad minima. | Unknown | No |
| 5 | Phases of learning corresponds to learning of initial conditions and regularization rule (differential equation) at different times. | False | No |
| 6 | Uniform initialization typically leads to approximation of straight line, while with another initialization rule bad behavior is often appears as approximation of zero. | Not Always True | No |

Explanation:

1. We think that it is true.
   a) This is the most convincing explanation.
   b) (2L, W) will be much more computationally expensive than (L, 2W).
2. This is true only if solution complexity increases. If it does increase (for example, in L-V system there could appear another cycle of dynamics if we increase T), then, obviously, neural network should be more complex too. But is this the only reason? Maybe there is a case when we increase T, and no new complexities pops up, but required network size increase too.
   a) Obviously true, but in the reverse way.
   b) We haven't work this out yet.
   c) We think that it is the most general cause of that behavior, but we don't know yet how to deal with "partial learning" and how to understand it.
3. This should be true because of 1a). Main question is - if we got not enought collocation points, is it really neccessary that result will always be zero function?
   a) We haven't work this out yet.
   b) We think that it is true.
4. This may be true because of general wisdom about noize in optimizer. Small batch size corresponds to a large noize in minimization direction, and we should carefully step towards the minima, but if we are using bigger batch size, then we are more confident about good minimization direction, so we can move towards it with larger steps. In the case of PINN we just don't know yet if this wisdom will stand. a) Same reason.
5. We think that this is not true. Our belief is that PINN learns particular properties of solution function, not just initial/boundary conditions.
6. This is because we think of bad and good initializations. Bad initializaion leads to approximation of straight line (not always on all domain), and if we are using uniform rule, this straight line may be non-zero. In the case of every other initialization, this line is typically zero function.

Problems to solve:

1. There are exactly three modes of learning: bad approximation (typically of straight line), slow approximation of solution, and good approximation. Sometimes second mode behaves like bad approximation - RMSE growing up, but neural network really learns some properties of solution function. It is hard to distinguish when to stop learning because of bad approximation or when to continue. How to resolve this problem?
   - Uniform initialization gives zero derivative almost everywhere, so neural network just cant learn anything about dynamics of system and learns straight line. Bad initialization leads to approximation of zero function.
2. Appropriate RMSE usually depend on problem statement. How should we deal with this problem in task #6?

## Our Roles

Daniil Laptev: tasks 1-3, 5-8.

Daniil Faryshev: task 4.

## Code Structure

Now we will describe the idea behind the structure of our repository and how files with code are structured.

For each problem in the table above there are a class related to it in the problems.py file. They are all organized in the following way:
- Attributes:
  1. Final time T, boundaries A and B, as float-point variables.
  2. Parameters of differential equation, such as diffusion coefficient etc. as float-point variables.
  3. Boundary values as PyTorch tensors. For ODE they are just values at time t = 0, for PDE they are PyTorch tensors with one tensor for each boundary (initial values are also boundary values on t = 0).
  4. Test points, equally spaced on the domain of equation, as PyTorch tensor.
  5. Numerical solution of the problem as PyTorch tensor.
- Methods:
  1. Method for boundary loss calculation.
  2. Method for regularization loss (PINN residue) calculation.
  3. Method for obtaining numerical solution for given problem statement.

And some other helpful elements.

In the utils.py file, there are implementation of Feed Forward neural networks in PyTorch and some helpful plotting functions. Function ```plot_losses``` is used to plot loss functions and RMSE in two subplots of one figure. Each plotting function has parameter "save", default False, that can be used to save figure in the images folder with the name provided; there are also "show" parameter, default True, that can be used to disable appearance of the plot (for example, when we want to visualize training process and save gif afterwards, we will use show=False and save=True).

Main .ipynb files for each problem we have yet analyzed contains:
- Training function.
- Plotting of the result.
- Plotting of the losses and errors.
- Experimentation with problem statements and searching for optimal hyperparameters.