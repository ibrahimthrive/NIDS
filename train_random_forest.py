"""
Step 2a: Random Forest Model - Strong Training & Evaluation
===========================================================
Trains a stronger Random Forest classifier for the 5 NSL-KDD classes.

Main improvements over the baseline:
  1. Uses stratified cross-validation and RandomizedSearchCV.
  2. Optimizes macro-F1, not only weighted F1, so rare classes matter.
  3. Oversamples minority classes in the training set.
  4. Tries balanced class-weight strategies.
  5. Reports weighted, macro, and per-class metrics.
  6. Saves the best model, search results, metrics JSON, and plots.

Note: A script can target >0.80 for accuracy, precision, recall, and F1, but no
code can honestly guarantee that every metric will exceed 0.80 on every split.
The printed target check will show which metrics/classes still need more work.
"""

import json
import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import CLASS_NAMES, run_preprocessing


ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = "plots"
RF_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "random_forest_model.pkl")
RF_SEARCH_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "rf_search_results.csv")
RF_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "rf_metrics.json")

RANDOM_STATE = 42
TARGET_SCORE = 0.80


def make_strong_class_weight(y_train):
    """Create stronger inverse-frequency weights for rare classes."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    # Push rare classes a little harder than standard balanced weights.
    counts = Counter(y_train)
    max_count = max(counts.values())
    for cls, count in counts.items():
        rarity_boost = np.sqrt(max_count / count)
        weight_dict[int(cls)] *= float(rarity_boost)

    return weight_dict


def oversample_minority_classes(X, y, target_ratio=0.35):
    """
    Randomly duplicate minority-class rows until each class has at least
    target_ratio * majority_count samples.
    """
    y_array = np.asarray(y)
    counts = Counter(y_array)
    majority_count = max(counts.values())
    target_count = int(majority_count * target_ratio)

    rng = np.random.default_rng(RANDOM_STATE)
    all_indices = [np.arange(len(y_array))]

    print("\nClass counts before oversampling:")
    for cls_idx, count in sorted(counts.items()):
        print(f"  {CLASS_NAMES[int(cls_idx)]:>6}: {count}")

    for cls_idx, count in sorted(counts.items()):
        if count >= target_count:
            continue
        cls_indices = np.where(y_array == cls_idx)[0]
        extra = rng.choice(cls_indices, size=target_count - count, replace=True)
        all_indices.append(extra)

    resampled_indices = np.concatenate(all_indices)
    rng.shuffle(resampled_indices)

    if hasattr(X, "iloc"):
        X_resampled = X.iloc[resampled_indices]
    else:
        X_resampled = X[resampled_indices]

    if hasattr(y, "iloc"):
        y_resampled = y.iloc[resampled_indices]
    else:
        y_resampled = y_array[resampled_indices]

    new_counts = Counter(np.asarray(y_resampled))
    print("\nClass counts after oversampling:")
    for cls_idx, count in sorted(new_counts.items()):
        print(f"  {CLASS_NAMES[int(cls_idx)]:>6}: {count}")

    return X_resampled, y_resampled


def build_search(y_train):
    """Build a deeper Random Forest hyperparameter search."""
    strong_weight = make_strong_class_weight(y_train)

    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [20, 30, 40],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.35, 0.5],
        "bootstrap": [True],
        "class_weight": ["balanced", "balanced_subsample", strong_weight],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_precision": make_scorer(precision_score, average="macro", zero_division=0),
        "macro_recall": make_scorer(recall_score, average="macro", zero_division=0),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "weighted_f1": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,   # outer RandomizedSearchCV already parallelizes across candidates/folds;
                    # n_jobs=-1 here too would oversubscribe the same cores
        verbose=0,
    )

    return RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=15,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True,
    )


def train_random_forest(X_train, y_train):
    """Train with oversampling plus CV hyperparameter search."""
    print("\nTraining stronger Random Forest...")
    X_fit, y_fit = oversample_minority_classes(X_train, y_train, target_ratio=0.35)

    search = build_search(y_fit)
    search.fit(X_fit, y_fit)

    print("\nBest CV result:")
    print(f"  macro-F1          : {search.best_score_:.4f}")
    print(f"  best parameters   : {search.best_params_}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_macro_f1"
        ).to_csv(RF_SEARCH_RESULTS_PATH, index=False)
        print(f"  Search results saved -> {RF_SEARCH_RESULTS_PATH}")
    except Exception as exc:
        print(f"  Could not save search results CSV: {exc}")

    return search.best_estimator_


def evaluate_model(model, X_test, y_test, label_names=CLASS_NAMES):
    """Compute weighted, macro, and per-class metrics."""
    print("\nEvaluating Random Forest on held-out test set...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "weighted_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["per_class"] = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in label_names
    }

    print("\n" + "=" * 62)
    for key, value in metrics.items():
        if key != "per_class":
            print(f"  {key.replace('_', ' ').title():<22}: {value:.4f}")
    print("=" * 62)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    print_target_check(metrics)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(RF_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {RF_METRICS_PATH}")

    return y_pred, metrics


def print_target_check(metrics):
    """Show whether the requested 0.80 target was met."""
    print(f"\nTarget check: each main score should be >= {TARGET_SCORE:.2f}")
    keys = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_f1"]
    for key in keys:
        status = "PASS" if metrics[key] >= TARGET_SCORE else "NEEDS WORK"
        print(f"  {key:<20} {metrics[key]:.4f}  {status}")

    print("\nPer-class F1 target check:")
    for cls_name, cls_metrics in metrics["per_class"].items():
        status = "PASS" if cls_metrics["f1"] >= TARGET_SCORE else "NEEDS WORK"
        print(f"  {cls_name:<8} F1={cls_metrics['f1']:.4f}  {status}")


def plot_confusion_matrix(y_test, y_pred, label_names=CLASS_NAMES, save_dir=PLOTS_DIR):
    """Plot and save count and normalized confusion matrices."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalized)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Random Forest - Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "rf_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved -> {path}")


def plot_feature_importances(model, feature_names, top_n=25, save_dir=PLOTS_DIR):
    """Save top feature importance plot."""
    os.makedirs(save_dir, exist_ok=True)
    importances = model.feature_importances_
    top_n = min(top_n, len(importances))
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(range(top_n), importances[indices], color="steelblue", edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    ax.set_xlabel("Feature")
    plt.tight_layout()
    path = os.path.join(save_dir, "rf_feature_importances.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved -> {path}")


def plot_per_class_metrics(metrics, label_names=CLASS_NAMES, save_dir=PLOTS_DIR):
    """Save grouped bar chart for per-class precision, recall, and F1."""
    os.makedirs(save_dir, exist_ok=True)
    precision = [metrics["per_class"][name]["precision"] for name in label_names]
    recall = [metrics["per_class"][name]["recall"] for name in label_names]
    f1 = [metrics["per_class"][name]["f1"] for name in label_names]

    x = np.arange(len(label_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    ax.bar(x, recall, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1, width, label="F1-Score", color="#C44E52")
    ax.axhline(TARGET_SCORE, color="#333333", linestyle="--", linewidth=1, label="0.80 target")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest - Per-Class Metrics", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "rf_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class metrics plot saved -> {path}")


def run_rf_pipeline(data):
    """End-to-end RF training and evaluation pipeline."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    selected_features = data["selected_features"]

    rf_model = train_random_forest(X_train, y_train)

    joblib.dump(rf_model, RF_MODEL_PATH)
    print(f"\nModel saved -> {RF_MODEL_PATH}")

    y_pred, metrics = evaluate_model(rf_model, X_test, y_test)

    print("\nGenerating evaluation plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importances(rf_model, selected_features)
    plot_per_class_metrics(metrics)

    return rf_model, metrics


if __name__ == "__main__":
    data = run_preprocessing(
        train_path="data/KDDTrain+.txt",
        test_path="data/KDDTest+.txt",
    )
    run_rf_pipeline(data)
