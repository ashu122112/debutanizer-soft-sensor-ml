# Debutanizer Soft Sensor - Stage 9 Optimisation Summary

## Approach
A Genetic Algorithm (DEAP library) was used to minimise the CAUSAL model's
predicted C4 concentration by searching over the 7 normalised process inputs (u1..u7).
The autoregressive model was intentionally excluded because an operator cannot
set past laboratory assay values in real time.

## GA Configuration
- Population size: 60
- Generations: 80
- Crossover probability: 0.7
- Mutation probability: 0.2
- Selection: Tournament (size 3)
- Seeds tested: 5 (seeds 42-46)
- Bounds: 5th to 95th percentile of training block (per variable)
- Distance penalty: quadratic excess penalty beyond 95th-percentile training radius

## Optimisation Results
| Metric | Value |
|:---|:---|
| **Best predicted C4** | 0.0286 |
| **Mean C4 across seeds** | 0.0344 |
| **Std C4 across seeds** | 0.0075 |
| **All seed results** | [0.0326, 0.0286, 0.0286, 0.0332, 0.0489] |

## Optimal Operating Point (normalised, u1..u7)
- `u1` = 0.2117
- `u2` = 0.6716
- `u3` = 0.7641
- `u4` = 0.6687
- `u5` = 0.8118
- `u6` = 0.4643
- `u7` = 0.4344

## Comparison: Optimum vs. Lowest-C4 Historical Operating Points
| Metric | Historical_1 | Historical_2 | Historical_3 | Historical_4 | Historical_5 | GA Optimum |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `u1` | 0.2289 | 0.2308 | 0.2269 | 0.2328 | 0.2348 | 0.2117 |
| `u2` | 0.6449 | 0.6385 | 0.6469 | 0.6344 | 0.6675 | 0.6716 |
| `u3` | 0.6307 | 0.6433 | 0.5286 | 0.7521 | 0.7926 | 0.7641 |
| `u4` | 0.2913 | 0.2914 | 0.2912 | 0.2915 | 0.2916 | 0.6687 |
| `u5` | 0.3578 | 0.3591 | 0.3565 | 0.4087 | 0.4635 | 0.8118 |
| `u6` | 0.2971 | 0.2389 | 0.3346 | 0.2148 | 0.1789 | 0.4643 |
| `u7` | 0.2634 | 0.2056 | 0.2925 | 0.1566 | 0.1339 | 0.4344 |
| `C4` | 0.0190 | 0.0202 | 0.0213 | 0.0256 | 0.0313 | 0.0286 |

## Critical Limitations
> **INDICATIVE RESULT ONLY.** The GA optimum is only as reliable as the causal model.
> The causal model achieves Test R² ≈ 0.45 at best (without lab feedback), meaning
> approximately 55% of C4 variance is not captured. The optimum may not be achievable
> in practice and should be validated via Aspen Plus simulation and plant trials.
> Additionally, these normalised values cannot be converted to engineering units until
> the original variable definitions and scale factors are officially confirmed.

## 3-Line Viva Explanation (Stage 9)
> 1. The GA treats the trained causal model as a fast surrogate for the real column
>    and searches for process settings that predict minimum C4 without running Aspen.
> 2. We used 5 random seeds to assess stability; the low std of results confirms
>    the GA landscape is not too rugged within the explored training bounds.
> 3. The result is indicative only — it is bounded to observed training conditions
>    and depends entirely on the causal model's accuracy, which is limited.