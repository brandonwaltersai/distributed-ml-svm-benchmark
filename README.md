# Distributed ML SVM Benchmark

![tests](https://github.com/brandonwaltersai/distributed-ml-svm-benchmark/actions/workflows/tests.yml/badge.svg)

A reproducible benchmark refactored from graduate machine-learning work, comparing a **distributed linear SVM in Apache Spark** with a **nonlinear RBF SVM in scikit-learn** under class imbalance.

The engineering question is more useful than simply asking which model has the highest accuracy:

> **When does distributed scale and training speed matter more than minority-class recall, and when is the slower nonlinear model worth the cost?**

## Verified historical results

The original experiment used an automotive-auction dataset with **72,000+ rows and 34 columns** and the binary target `IsBadBuy`.

| Model | Training time | Accuracy | Balanced accuracy | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Spark MLlib LinearSVC | ~3 min | **0.8218** | 0.5918 | 0.2869 | 0.2834 |
| scikit-learn RBF SVC | ~43 min | 0.6447 | **0.6449** | **0.6451** | **0.3087** |

The Spark model trained roughly **14× faster** and achieved much higher raw accuracy. The RBF model was substantially slower but recovered far more of the minority class and produced better balanced accuracy and F1.

That is the core result: **the best model depends on the operational objective, not a single headline metric.**

## What this repository contains

This repository is a cleaned, reusable implementation derived from the methodology of the original experiment. It is **not** presented as the untouched course submission.

- `src/data_prep.py` — shared dataset loading, target validation, and train/test splitting
- `src/sklearn_svm.py` — RBF SVM pipeline with imputation, scaling, one-hot encoding, optional grid search, and threshold evaluation
- `src/spark_svm.py` — Spark MLlib LinearSVC pipeline with numeric/categorical preprocessing
- `src/metrics.py` — consistent evaluation helpers
- `docs/results.md` — verified historical results and engineering interpretation
- `tests/` — lightweight tests that do not require the original course dataset
- `.github/workflows/tests.yml` — CI for the reusable benchmark code

## Why accuracy is misleading here

`IsBadBuy` is imbalanced. A model can look strong on raw accuracy while missing many positive cases. For that reason, the comparison emphasizes:

- balanced accuracy
- minority-class recall
- F1
- training time / scalability

The Spark model's 0.8218 accuracy is therefore not treated as an automatic win. Its recall of 0.2869 means it missed a large share of positive cases. The nonlinear model traded runtime and raw accuracy for materially better minority-class detection.

## Reproducing with your own compatible dataset

The original course dataset is **not redistributed here**. Provide a CSV with a binary target named `IsBadBuy` (or adapt the CLI target argument) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the scikit-learn path:

```bash
python -m src.sklearn_svm --data /path/to/data.csv --target IsBadBuy
```

Run the Spark path:

```bash
python -m src.spark_svm --data /path/to/data.csv --target IsBadBuy
```

Exact historical metrics in `docs/results.md` come from the original graduate experiment and should not be expected to reproduce without the same source dataset, split, environment, and tuning choices.

## Engineering takeaways

1. **Distributed training can radically reduce wall-clock time**, but the fastest model may not meet the business objective.
2. **Imbalanced classification requires metrics beyond accuracy.** Balanced accuracy, recall, and F1 changed the interpretation of the result.
3. **Thresholds and model family matter.** The nonlinear RBF model captured minority cases much better, at significant compute cost.
4. **Infrastructure context matters.** Spark is attractive when scale and parallelism dominate; a nonlinear single-node model can still be preferable when minority-class capture is more important than throughput.

## Provenance and limitations

This repository was refactored for portfolio use from Brandon Walters's DATA 660 graduate work. The original written analysis is retained privately; instructor material, course PDFs, and the source dataset are not redistributed here. Historical metrics are preserved as evidence from that experiment, while the public code is a reusable refactoring rather than a claim that the exact original environment is fully reproducible.

## Stack

Python · Apache Spark / PySpark · scikit-learn · pandas · NumPy

## Author

Brandon Walters — [LinkedIn](https://www.linkedin.com/in/bw172b29208/) · [GitHub](https://github.com/brandonwaltersai)
