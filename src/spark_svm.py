from __future__ import annotations

import argparse
import json

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType


def build_pipeline(df, target: str = "IsBadBuy", reg_param: float = 0.1, max_iter: int = 200):
    feature_cols = [c for c in df.columns if c != target]
    numeric = [f.name for f in df.schema.fields if f.name in feature_cols and isinstance(f.dataType, NumericType)]
    categorical = [c for c in feature_cols if c not in numeric]

    stages = []
    assembled_inputs = []

    if numeric:
        imputed = [f"{c}__imputed" for c in numeric]
        stages.append(Imputer(inputCols=numeric, outputCols=imputed, strategy="median"))
        assembled_inputs.extend(imputed)

    if categorical:
        indexed = [f"{c}__idx" for c in categorical]
        encoded = [f"{c}__ohe" for c in categorical]
        for c, out in zip(categorical, indexed):
            stages.append(StringIndexer(inputCol=c, outputCol=out, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=indexed, outputCols=encoded, handleInvalid="keep"))
        assembled_inputs.extend(encoded)

    stages.append(VectorAssembler(inputCols=assembled_inputs, outputCol="features_raw", handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True))
    stages.append(StringIndexer(inputCol=target, outputCol="label", handleInvalid="error"))
    stages.append(LinearSVC(featuresCol="features", labelCol="label", regParam=reg_param, maxIter=max_iter))
    return Pipeline(stages=stages)


def run(path: str, target: str = "IsBadBuy", reg_param: float = 0.1, max_iter: int = 200) -> dict[str, float]:
    spark = SparkSession.builder.appName("distributed-ml-svm-benchmark").getOrCreate()
    try:
        df = spark.read.option("header", True).option("inferSchema", True).csv(path).dropna(subset=[target])
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        model = build_pipeline(train, target, reg_param, max_iter).fit(train)
        pred = model.transform(test).cache()

        def score(metric_name: str) -> float:
            return MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName=metric_name
            ).evaluate(pred)

        # Spark exposes weighted precision/recall/F1 directly; balanced accuracy
        # is computed from per-class recall below.
        recalls = {}
        for label in (0.0, 1.0):
            subset = pred.filter(pred.label == label)
            total = subset.count()
            correct = subset.filter(subset.prediction == label).count()
            recalls[label] = correct / total if total else 0.0

        return {
            "accuracy": score("accuracy"),
            "balanced_accuracy": (recalls[0.0] + recalls[1.0]) / 2,
            "weighted_precision": score("weightedPrecision"),
            "weighted_recall": score("weightedRecall"),
            "f1": score("f1"),
        }
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Spark MLlib LinearSVC on a binary CSV dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="IsBadBuy")
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.target, args.reg_param, args.max_iter), indent=2))


if __name__ == "__main__":
    main()
