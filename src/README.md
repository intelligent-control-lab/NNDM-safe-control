# Safe control with Neural Network Dynamic Models

This is the code base for L4DC 2022 submission: Safe control with Neural Network Dynamic Models.

The algorithm depends on a modified version of [NeuralVerification](https://github.com/intelligent-control-lab/NeuralVerification.jl/tree/nn-safe-control). Please make sure this package is installed properly.

### Trajectory tracking
To compare different solvers for tracking with NNDM, run `exp_tracking.ipynb`

### Trajectory tracking with safety constraints.
To reproduce the collision avoidance results, run `collision_index_learning.ipynb`.
To reproduce the safe following results, run `following_index_learning.ipynb`.

### Scalability and Tracking Plots
To study the scalability of MIND, or plot tracking trajectories and control signals, run `scalability_and_tracking_plots.ipynb`