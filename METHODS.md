# Technical Methods and Justification

This document provides the academic and technical grounding for the BBO capstone strategy. It is intended as a reference for anyone reviewing the repository who wants to understand not just what was built but why specific choices were made.

---

## Surrogate Model: Gaussian Process Regression

The surrogate model used throughout this project is a **Gaussian Process (GP) regressor** with a **Matern 5/2 kernel**, implemented via scikit-learn's `GaussianProcessRegressor`.

### Why GPs?

GPs are the canonical surrogate model for expensive black-box optimization because they:
- Are non-parametric and make no strong assumptions about the function's form
- Naturally quantify predictive uncertainty alongside the mean prediction
- Produce well-calibrated uncertainty estimates even with very small datasets (10-45 points)
- Allow the acquisition function to balance exploration and exploitation in a principled way

**Key reference:** Brochu, E., Cora, V. M., and de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning.* arXiv:1012.2599.

### Why Matern 5/2?

The Matern 5/2 kernel assumes the unknown function is twice differentiable. This is a standard prior for real-world processes that are smooth but not infinitely so. It avoids the over-smoothing of the RBF (squared exponential) kernel, which can cause the GP to miss sharp peaks, while producing better interpolation than the Matern 3/2.

---

## Acquisition Function: Upper Confidence Bound (UCB)

The acquisition function used to select each query point is **UCB**, scored as:

```
UCB(x) = mu(x) + kappa * sigma(x)
```

where `mu(x)` is the GP predicted mean, `sigma(x)` is the predicted standard deviation, and `kappa` controls exploration vs exploitation.

### Why UCB over Expected Improvement (EI)?

UCB was chosen over EI because kappa is an explicit, interpretable hyperparameter that can be tuned per function per round. EI implicitly encodes a fixed exploration preference that cannot be adjusted without modifying the acquisition function itself. The ability to raise or lower kappa based on observed performance history has been the core adaptive mechanism of this project.

**Key reference:** Snoek, J., Larochelle, H., and Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS 2012. arXiv:1206.2944.

---

## Adaptive Kappa Schedule

Rather than fixing kappa across all functions and rounds, kappa is assigned per function based on performance history each week:

| kappa | When applied |
|---|---|
| 2.576 | Round 1 -- high uncertainty, no evidence yet |
| 2.5 | Functions that have stalled or regressed |
| 2.0 | Balanced cases with mixed recent results |
| 1.5 | Functions with a recent new best (transition phase) |
| 1.0 | Functions improving consistently |
| 0.5 | Functions with a breakthrough or new best every round |

This decaying exploration schedule is grounded in the theoretical result that reducing the UCB confidence bound over time yields sublinear cumulative regret.

**Key reference:** Srinivas, N., Krause, A., Kakade, S., and Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design.* ICML 2010. arXiv:0912.3995.

---

## Special Strategy: Function 1 (Max-Distance Grid Sweep)

Function 1 has returned near-zero outputs across all rounds. With no meaningful signal, the GP has nothing to learn from and its acquisition scores are arbitrary. A **max-distance grid sweep** is used instead:

1. A 200x200 grid is laid over [0.05, 0.95]^2
2. For each grid point, the minimum distance to all previously observed inputs is computed using `scipy.spatial.distance.cdist`
3. The grid point with the largest minimum distance is selected

This guarantees that each query covers a genuinely new region rather than clustering near prior observations.

---

## Libraries

| Library | Version | Role |
|---|---|---|
| scikit-learn | latest | GaussianProcessRegressor, Matern kernel |
| NumPy | latest | Array operations, candidate generation |
| SciPy | latest | cdist for max-distance query (F1) |
| Matplotlib | latest | GP surface visualisations |
| Pandas | latest | Progress tracking tables |

### Considered but not adopted

- **GPyTorch**: GPU-accelerated GP inference. Would be appropriate if the query budget were extended to hundreds of points per function. At current sample sizes (15-45 points), the engineering overhead is not justified.
- **BoTorch**: PyTorch-based BBO library from Meta with advanced acquisition functions (Knowledge Gradient, batch UCB). Worth revisiting in final rounds or for future extensions.
- **PyTorch / TensorFlow**: Neural network surrogates require substantially more data than is available here to avoid overfitting. The GP provides better uncertainty quantification at this sample scale.

---

## Benchmarking and Future References

- **HPOBench**: AutoML benchmarking suite providing empirical baselines for GP-UCB on hyperparameter optimization tasks. Would allow contextualising capstone results against published benchmarks.
- **Spearmint** (Snoek et al.): The original open-source implementation of practical Bayesian optimization, useful as a reference implementation.
- **Optuna / Hyperopt**: Production-grade BBO libraries with broader acquisition function support, relevant for understanding how the approach here compares to industry-standard tooling.

---

*This document is updated alongside the weekly notebooks. Last updated: Round 6.*
