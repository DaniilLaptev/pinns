# pinns
Physics-Informed toolkit for solving physics-informed tasks.

---

This is a framework that will allow user to train Physics-Informed Neural Networks (and not only them) to solve classic machine learning tasks or differential equations, learn them from data, make discoveries using a wide range of tools for training and analysis.

Please note that this project is still in the prototyping stage, so the lack of documentation is due to the incomplete structure of the project, and the signatures of the methods must change, the classes and their functionality will also change, etc. Comments with reasonable explanations will be added when everything is settled.

## Future work

- Different adaptive samplers will be added.
- Smart hyperparameters tuning methods for our setting. User is still able to apply Optuna, for example, but it is highly desirable to reduce amount of written code since this is not what researcher that will use our framework wants.
- Default benchmarks will be added as examples (Navier-Stokes, Schrodinger). Perhaps with a numerical solvers that will allow user to change parameters and boundary conditions in one place, but it will take longer to implement and is low priority task.
- Adaptive activation functions and some non-adaptive exotic AF (such as dynamic ReLU) will be added.
- Much more optimizers will be added, including evolutionary approaches.
- An Analyzers toolkit: Information Propagation, Spectral Bias, Neural Tangent Kernel, loss landscape etc.
- Different types of PINNs will be implemented. 
- A toolkit for solving a large diversity of inverse problems.
- I'm looking forward into including a large amount of probabilistic methods into our library (BNNs, GPs, bayesian optimization for parameters, uncertainty quantification etc.)
- An interpretation toolkit: derivation of closed-form mathematical expression of solution, analysis of learned features ("zoom" into the model weights and their structure) and other forms of explanation and interpretation of trained model.
- I hope that Neural Operators will be somehow related to instruments that we will have in this framework. Since they are implemented in DeepXDE and produce interesting research - we will implement them in future.
- Maybe something else.