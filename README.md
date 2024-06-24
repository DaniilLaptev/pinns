# pinns
Project where I study PINN methodology for solving various differential equations.

---

### Important features:

- Add more examples. Each example should focus on one important feature, and for each example there should be different DE.
- Add more different problems: ODEs of higher order, many-dimensional ODEs and PDEs, classic benchmark tasks (KdV, Burgers, N-S, Schrodinger, Helmholtz etc.), stochastic DEs.
- Add interface for different callbacks: saving model, plotting predictions, calculating some statistics etc.
- Add interface for early stopping: by nan/inf, by epsilon, by some user-defined or predefined criteria (for example, when loss does not change).
- Add some more opimizers from different sources (PyTorch, pymoo, scipy.optimize etc.) and test them.
- Add dynamic methods: change loss coefficients during training, samplers, optimizer etc.
- Add many different activation functions that are not in PyTorch. Add adaptive activation functions.
- Add adaptive samplers. Maybe it is more appropriate to make a sampler with some parameters (for example, sample from beta distribution) and use dynamic methods or callbacks to change parameters of that sampler during training.
- Add different PINNs that are already discussed in literature: xPINNs, cPINNs, fPINNs, GP-PINNs, GPT-PINNs, Bayesian, GANs, PhyLSTM, PhyGeoNet, CNNs. Maybe implement MAMBA-like model and more exotic models, if feasible.
- If loss decreases, it is not clear if PINN actually approximates solution. How can we be sure that training is not broken and loss does not fool us? Maybe we should calculate difference between model predictions and stop if they are not changing, but loss does, and add this strategy into early stopping.
- Add pinns.inverse module for inverse problems.
- Add pinns.Tuner module for automatic tuning of various hyperparameters, meta-learning, pareto frontier derivation, hyperparameters importance analysis and more.
- Add an example of how to define hard constraints. Restructure model prediction logic so user will edit only predict(x) method.
- Make autograd derivative much more faster. Maybe use torch.func. Benchmark different methods (maybe for large number of input and output variables jacobian will be faster than batched grad).

### Not so important features:

- Add fancytensor that will let user access variables not by slicing, but also by their name.
- Add commentaries and documentation.
- Define more different activation functions.
- Add method to obtain learning frontier.
- Add many-dimensional finite differences derivative computation.
- Add fractional derivative computation.
- Make available training on multiple devices.
- Add pinns.interpret module, which will provide methods for deriving analytical formula, examining of learned representations, weights analysis etc.
- Add pinns.analyze module. There will be different methods for analyzing training process of PINNs: NTK, spectral bias, information propagation, loss landscape analysis, learning frontier monitoring etc.
- Add an example of sequential learning: at some iterations, we use learned model to predict target function values (at some points b), and add predictions into the constraints data. This operation must use information about learning frontier.
- Build unifying interface for different Kolmogorov-Arnold Networks, experiment with them and choose the most appropriate ones for further investigation.
- Maybe we should make various predefined trainers for different problems? For example, TorchTrainer will expect pytorch-like workflow, EvolutionaryTrainer and BayesianTrainer will expect evolutionary strategies and bayesian optimization logic respectively; trainers library may include some kind of task-specific features.

### Future experiments:

- Compare different optimizers.
- Compare different models.
- Compare different learning methods.
- Check if information propagation present in all of PINN tasks. Moreover, check if it present in different optimizers and GP-PINNs.
- What if we use transformer encoder architecture, will its attention mechanism be able to capture the relationships between input variables? For models with context, they are may be a room for experimenting with scalability of data and pre-trained PINNs. For example, if we want to encode some more variables along with default ones. Moreover, they may be a possibility of that if we somehow can encode differential equation and pass it into the model (maybe create something like token library, but for differential operators, or just pass them as text).