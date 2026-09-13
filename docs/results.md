# Historical Benchmark Results

These results come from the original DATA 660 graduate experiment that motivated this public refactor. The public repository does not redistribute the course dataset, so these values are preserved as **historical experiment results**, not as claims that the current refactored code has reproduced the exact run.

## Dataset and task

- Automotive auction records
- More than 72,000 rows
- 34 columns
- Binary target: `IsBadBuy`
- Preprocessing included median imputation, numeric standardization, and categorical one-hot encoding
- Evaluation emphasized class imbalance rather than raw accuracy alone

## Compared models

### Apache Spark MLlib LinearSVC

Two configurations were evaluated in the original work:

- `regParam=0.0`, `maxIter=100`
- `regParam=0.1`, `maxIter=200`

Best reported result:

| Metric | Result |
|---|---:|
| Accuracy | 0.8218 |
| Balanced accuracy | 0.5918 |
| Precision | 0.2800 |
| Recall | 0.2869 |
| F1 | 0.2834 |
| Training time | ~3 min |

### scikit-learn RBF SVC

The original workflow included a baseline model, parameter search, and threshold adjustment.

Best reported result:

| Metric | Result |
|---|---:|
| Accuracy | 0.6447 |
| Balanced accuracy | 0.6449 |
| Precision | 0.2029 |
| Recall | 0.6451 |
| F1 | 0.3087 |
| Training time | ~43 min |

## Engineering interpretation

The models optimized different operational properties.

**Spark LinearSVC** was approximately 14× faster and produced much higher raw accuracy. That makes it attractive when throughput, distributed execution, and scale are primary constraints. However, minority-class recall was low, so the strong accuracy number masked weak positive-case capture.

**RBF SVC** was much slower and had lower raw accuracy, but it materially improved minority-class recall and balanced accuracy. In a workflow where missing positive cases is costly, that trade can be justified.

The main lesson is therefore not "Spark wins" or "RBF wins." It is that **model selection should be tied to operational cost, class imbalance, and the consequence of false negatives—not a single aggregate metric.**

## Reproducibility note

The original course dataset and instructor materials are not included in this repository. Exact reproduction of these historical values requires the same source data, split, environment, preprocessing, hyperparameters, and threshold choices used in the original run. The code in `src/` is a portfolio refactor of that methodology for use with a compatible user-supplied dataset.
