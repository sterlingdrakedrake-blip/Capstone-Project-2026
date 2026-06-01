# Model Card: Gaussian Process UCB Optimizer

This model card follows the Model Cards for Model Reporting framework (Mitchell et al., 2019) and documents the Bayesian optimization approach used in the BBO Capstone project.

---

## Model Details

**Model name:** GP-UCB Black-Box Optimizer

**Version:** Round 10 (adaptive kappa schedule)

**Model type:** Gaussian Process surrogate with Upper Confidence Bound acquisition function

**Developed by:** BBO Capstone project participant

**Date:** Academic year 2025-2026

**License:** Academic use only

**Contact:** Available via GitHub repository

---

## Intended Use

**Primary intended use:**
Sequential optimization of eight unknown black-box functions over [0, 1]^d input spaces, with one evaluation per function per week. The model selects the most informative next query point at each round.

**Primary intended users:**
The project participant, course assessors, and anyone reviewing the methodology for academic or educational purposes.

**Out-of-scope uses:**
- Real-time or batch optimization requiring more than one query per iteration.
- Production deployment in safety-critical systems.
- Functions with categorical, integer, or constrained inputs.
- Problems requiring multi-objective optimization.

---

## Factors

**Relevant factors:**
- **Function dimensionality:** The approach is applied to 2D through 8D functions. GP performance degrades as dimensionality increases due to the curse of dimensionality.
- **Budget:** Results depend heavily on the query budget. With 9 rounds completed, higher-dimensional functions are severely undersampled.
- **Function smoothness:** The Matern 5/2 kernel assumes twice-differentiable functions. Performance degrades on functions with discontinuities or isolated spikes.
- **Noise level:** Some functions appear stochastic. Noisy outputs reduce GP accuracy and may mislead the acquisition function.

---

## Metrics

**Performance metrics used:**

| Metric | Description |
|---|---|
| Best-found output | Maximum output value observed across all rounds for each function |
| Round-over-round improvement | Whether each new query produced a new best |
| GP predictive accuracy | Qualitative -- assessed by whether the acquisition function recommendations correlate with observed improvements |

**Performance summary as of Round 10:**

| Function | Initial Best | Best Found | Improvement | Status |
|---|---|---|---|---|
| F1 | 0.0000 | 0.0000 | 0% | No signal found |
| F2 | 0.6112 | 0.6467 | +5.8% | New best R9 |
| F3 | -0.0348 | -0.0106 | +69.5% | Best from R2 |
| F4 | -4.0255 | +0.3657 | Positive | Breakthrough R4 |
| F5 | 1088.86 | 6445.99 | +492% | Strong consistent gains |
| F6 | -0.7143 | -0.2138 | +70.1% | Best from R4 |
| F7 | 1.3650 | 1.9584 | +43.5% | Best from R2 |
| F8 | 9.5985 | 9.9795 | +4.0% | New best R9 |

---

## Training Data

The GP surrogate is fitted at each round to all accumulated (input, output) pairs for each function. There is no separate training/test split -- the model is updated online with each new observation. See DATASHEET.md for full dataset documentation.

---

## Evaluation Data

The model is evaluated implicitly through the optimization loop: a better query choice yields a higher output value. There is no held-out evaluation set.

---

## Technical Specifications

**Surrogate model:** `sklearn.gaussian_process.GaussianProcessRegressor`

**Kernel:** Matern 5/2 (`nu=2.5`), with length-scale and noise parameters optimized by maximizing marginal likelihood at each round (`n_restarts_optimizer=10`)

**Output normalization:** `normalize_y=True` -- outputs are standardized before fitting to prevent scale dominance

**Acquisition function:** Upper Confidence Bound (UCB)
```
UCB(x) = mu(x) + kappa * sigma(x)
```

**Candidate generation:** 100,000 points drawn uniformly from [0, 1]^d at each round

**Adaptive kappa schedule:**

| kappa | When applied |
|---|---|
| 0.5 | Functions with new best or consistent gains -- exploit very hard |
| 1.0 | Functions with recent positive result -- exploit moderately |
| 1.5 | Functions transitioning from exploration to exploitation |
| 2.0 | Balanced cases with mixed recent results |
| 2.5 | Functions that have stalled or regressed -- explore new regions |

**Special case -- Function 1:**
After 9 rounds of near-zero outputs, the GP pipeline was replaced with a pre-planned systematic grid sweep. Four target coordinates were selected by iteratively maximizing the minimum distance to all previously observed points across a 300x300 grid over [0.05, 0.95]^2.

---

## Caveats and Recommendations

**Known limitations:**

1. **Cubic scaling:** `GaussianProcessRegressor` uses Cholesky decomposition, which scales as O(n^3). This is not a practical constraint at current dataset sizes but would become limiting beyond ~500 points per function.

2. **Single-query budget:** The one-query-per-week constraint means a single misleading result can misdirect strategy for multiple rounds. Standard Bayesian optimization assumes batch queries to reduce this variance.

3. **Smoothness assumption:** The Matern 5/2 kernel assumes twice-differentiable functions. Functions with narrow isolated peaks (plausibly F1) are not well-modeled.

4. **Local optima:** With limited data, the GP may converge confidently around a local rather than global maximum. F4's breakthrough in Round 4 has not been reproducible in subsequent rounds, suggesting the positive basin may be narrow and the GP is unable to reliably relocate it.

5. **Dimensionality:** Results for F7 (6D) and F8 (8D) should be interpreted cautiously. 39 and 49 observations respectively provide minimal coverage of a 6D and 8D unit hypercube.

**Recommendations for future use:**
- For datasets larger than ~200 points, consider GPyTorch for GPU-accelerated inference.
- For higher-dimensional functions, Latin hypercube sampling or Sobol sequences would provide better initial coverage than random uniform sampling.
- BoTorch's Knowledge Gradient or batch UCB acquisition functions would be more sample-efficient with a larger query budget.

---

## Ethical Considerations

The model is used exclusively for academic optimization of synthetic functions. No personal data is processed, no real-world decisions are made automatically, and no individuals are affected by the model's outputs. The optimization targets (drug side effects, warehouse placement, etc.) are simulations for educational purposes only.

---

## References

- Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS 2012. arXiv:1206.2944.
- Brochu, E., Cora, V. M., and de Freitas, N. (2010). A Tutorial on Bayesian Optimization of Expensive Cost Functions. arXiv:1012.2599.
- Srinivas, N., Krause, A., Kakade, S., and Seeger, M. (2010). Gaussian Process Optimization in the Bandit Setting. ICML 2010. arXiv:0912.3995.
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAccT 2019.

---

*Model card version: Round 10. Last updated following Week 9 query results.*
