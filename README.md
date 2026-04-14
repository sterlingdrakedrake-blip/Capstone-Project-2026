# Black-Box Optimisation Capstone

## Section 1: Project Overview

This project is a Bayesian optimisation competition in which the goal is to find the maximum output of eight unknown (black-box) functions through a series of iterative queries. Each function accepts a fixed-dimensional input vector and returns a single scalar output. The internal mechanics of every function are hidden; only the input-output relationship is observable through the data.

The project simulates a class of problems that appear frequently in real-world machine learning and engineering: situations where evaluating a function is expensive, slow, or otherwise limited, and where the structure of the objective is not known in advance. Examples include hyperparameter tuning for deep learning models, drug candidate screening, materials science experiments, and industrial process optimisation. In each of these settings, a practitioner cannot afford to evaluate every possible configuration and must instead make intelligent, data-driven decisions about where to look next.

From a career perspective, the skills developed here transfer directly to applied ML roles. The ability to reason under uncertainty, build surrogate models, balance exploration against exploitation, and revise a strategy based on new evidence is central to any data science or MLOps function. Many production ML workflows involve exactly this kind of sequential decision-making, whether tuning a recommendation system, calibrating a simulation, or optimising a business process where full enumeration is not feasible.

---

## Section 2: Inputs and Outputs

Each of the eight functions takes a continuous input vector with values constrained to the unit interval [0, 1] and returns a single continuous scalar output. The dimensionality increases across functions, from 2D up to 8D.

**Input format:** `x1-x2-x3-...-xn`
Each value must begin with `0` and be specified to six decimal places.

**Examples:**
```
Function 1 (2D):  0.718071-0.879689
Function 3 (3D):  0.611094-0.382817-0.600071
Function 5 (4D):  0.524778-0.842229-0.984007-0.984075
Function 7 (6D):  0.007196-0.287182-0.431297-0.103298-0.259892-0.771925
Function 8 (8D):  0.024544-0.175956-0.116596-0.359046-0.449942-0.482790-0.135292-0.369405
```

**Output:** A single floating-point scalar returned by the portal after each submission. This value represents the function's response at the queried input and is the signal used to update the surrogate model.

**Function reference table:**

| Function | Dims | Initial Data Points | Simulated Domain |
|---|---|---|---|
| F1 | 2 | 10 | Radiation source detection |
| F2 | 2 | 10 | Noisy ML log-likelihood surface |
| F3 | 3 | 15 | Drug discovery - adverse reaction minimisation |
| F4 | 4 | 30 | Warehouse placement optimisation |
| F5 | 4 | 20 | Chemical process yield |
| F6 | 5 | 20 | Cake recipe multi-objective scoring |
| F7 | 6 | 30 | ML model hyperparameter tuning |
| F8 | 8 | 40 | High-dimensional model performance optimisation |

---

## Section 3: Challenge Objectives

The goal for every function is **maximisation**: find the input vector that produces the highest possible output value. Some functions are framed as minimisation problems in their original domain (for example, minimising adverse drug reactions or minimising recipe cost), but these are transformed so that maximising the returned output is always the correct objective.

**Constraints and limitations to consider:**

- **One query per function per week.** Evaluations are expensive by design; each submission returns a single new data point. Strategy must account for this scarcity.
- **No access to the function internals.** Gradients, structure, and domain knowledge are all unavailable. The only information is the growing set of (input, output) pairs.
- **Unknown noise levels.** Some functions appear noisy (F2 in particular), meaning the same input queried twice might return slightly different outputs. This complicates exploitation.
- **Unknown number of optima.** Functions may be unimodal or multimodal. A strong local result does not guarantee it is the global maximum.
- **Delayed feedback.** Results are returned on a weekly cycle, not immediately. Batch or lookahead strategies that would benefit from rapid iteration are not applicable here.
- **Dimensionality varies.** Higher-dimensional functions (F7, F8) require more data to build an accurate surrogate and are more susceptible to the curse of dimensionality.

---

## Section 4: Technical Approach

### Surrogate Model

All queries are generated using a **Gaussian Process (GP) regressor** with a Matern 5/2 kernel, implemented via scikit-learn's `GaussianProcessRegressor`. The GP is fitted to all available (input, output) pairs for a given function and produces a predicted mean and uncertainty estimate for any candidate point in the input space. The kernel's length-scale and noise parameters are optimised by maximising the marginal likelihood at each round.

### Acquisition Function

The **Upper Confidence Bound (UCB)** acquisition function is used to score candidate points:

```
UCB(x) = mu(x) + kappa * sigma(x)
```

where `mu(x)` is the GP predicted mean, `sigma(x)` is the predicted standard deviation, and `kappa` controls the exploration-exploitation trade-off. A pool of 100,000 random candidates is drawn uniformly from [0, 1]^d at each round and the highest-scoring point is submitted.

### Adaptive Kappa Strategy

Rather than fixing `kappa` across all functions and all rounds, `kappa` is tuned per function based on observed performance:

| kappa value | When applied | Rationale |
|---|---|---|
| 2.576 | Round 1 (all functions) | High uncertainty, prioritise exploration |
| 2.5 | Functions that have regressed or stalled | Force the GP to search new regions |
| 2.0 | Balanced cases with mixed results | Equal weight to mean and uncertainty |
| 1.5 | Functions with a recent new best | Begin exploiting the promising region |
| 1.0 | Functions improving consistently every round | Concentrate search near the current peak |

This schedule implements a principled **decay from exploration to exploitation** as evidence accumulates, rather than applying a single heuristic blindly.

### Special Case: Function 1

Function 1 has returned near-zero outputs across all three rounds. With no meaningful signal, the GP has nothing to fit and its recommendations are unreliable. For this function a **max-distance grid sweep** is used instead: a fine grid of 200 x 200 points is laid over [0.01, 0.99]^2 and the point furthest from all previously observed inputs is selected. This guarantees coverage of unvisited regions without relying on a model that has no information to work with.

### Progress After Four Rounds

| Function | Initial Best | After R1 | After R2 | After R3 | After R4 (best so far) | Trend |
|---|---|---|---|---|---|---|
| F1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | TBC | No positive signal yet |
| F2 | 0.6112 | 0.6112 | 0.6112 | 0.6112 | TBC | Matched but not beaten |
| F3 | -0.0348 | -0.0348 | -0.0106 | -0.0106 | TBC | New best found R2 |
| F4 | -4.0255 | -1.3242 | -1.3242 | -1.3242 | TBC | Best from R1, regressing |
| F5 | 1088.86 | 2021.56 | 2201.35 | 2761.25 | TBC | New best every round |
| F6 | -0.7143 | -0.7143 | -0.7143 | -0.4256 | TBC | New best R3 |
| F7 | 1.3650 | 1.3650 | 1.9584 | 1.9584 | TBC | New best R2 |
| F8 | 9.5985 | 9.8100 | 9.8527 | 9.8684 | TBC | Steady gains every round |

### Considered Alternatives

- **SVM classifier:** A soft-margin SVM with an RBF kernel could classify input regions as high or low performing (e.g. above/below the top quartile). This becomes more viable as data accumulates. At current sample sizes (12 to 42 points per function), overfitting is a significant risk.
- **Linear/logistic regression:** Useful as a quick feature relevance check via correlation analysis, which is run each round before querying. Not used for query selection due to the non-linear response surfaces observed in most functions.
- **Random search:** Used as a fallback baseline. The 100,000-candidate random pool in the UCB step effectively incorporates random search as a component, while the GP acquisition function filters it intelligently.

### Repository Structure

```
|-- data/
|   |-- function_1_initial_inputs.npy
|   |-- function_1_initial_outputs.npy
|   `-- ...                            # initial data for all 8 functions
|-- week1_bayesian_optimisation.ipynb  # Round 1 queries and EDA
|-- week2_bayesian_optimisation.ipynb  # Round 2 - adaptive kappa introduced
|-- week3_bayesian_optimisation.ipynb  # Round 3 - exploit-focused, F1 fix
`-- README.md
```

> **Note:** This README is a living document and will be updated after each submission round to reflect new results, strategy changes, and any shifts in the approach.
