"""
Step 3: Model Comparison
=========================
Loads both trained models, generates a side-by-side metrics comparison,
and produces a combined visualisation report.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from preprocessing import CLASS_NAMES
# FocalLoss must be imported so its @register_keras_serializable decorator
# runs before any load_model call, enabling automatic deserialisation.
from train_lstm import FocalLoss  # noqa: F401

ARTIFACTS_DIR  = "artifacts"
PLOTS_DIR      = "plots"
RF_MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "random_forest_model.pkl")
LSTM_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "lstm_model", "lstm_final.keras")


def load_models_and_data():
    """Load both models and test data from saved artifacts."""
    print("📦 Loading models and artifacts...")
    rf_model   = joblib.load(RF_MODEL_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_DIR)

    scaler      = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    le          = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
    top_indices = joblib.load(os.path.join(ARTIFACTS_DIR, "selected_feature_indices.pkl"))

    return rf_model, lstm_model, scaler, le, top_indices


def get_predictions(rf_model, lstm_model, X_test, y_test):
    """Generate predictions from both models."""
    y_pred_rf   = rf_model.predict(X_test)
    X_test_3d   = X_test.reshape(-1, 1, X_test.shape[1])
    y_prob_lstm = lstm_model.predict(X_test_3d, verbose=0)
    y_pred_lstm = np.argmax(y_prob_lstm, axis=1)
    return y_pred_rf, y_pred_lstm


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_comparison_bar(rf_metrics, lstm_metrics, save_dir=PLOTS_DIR):
    """Side-by-side bar chart comparing RF vs LSTM on all metrics."""
    os.makedirs(save_dir, exist_ok=True)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    rf_vals   = [rf_metrics["accuracy"],   rf_metrics["precision"],
                 rf_metrics["recall"],     rf_metrics["f1"]]
    lstm_vals = [lstm_metrics["accuracy"], lstm_metrics["precision"],
                 lstm_metrics["recall"],   lstm_metrics["f1"]]

    x     = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, rf_vals,   width, label="Random Forest", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, lstm_vals, width, label="LSTM",          color="#C44E52", edgecolor="white")

    # Value labels on bars
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Random Forest vs LSTM", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Comparison bar chart saved → {path}")


def plot_dual_confusion_matrices(y_test, y_pred_rf, y_pred_lstm,
                                  label_names=CLASS_NAMES, save_dir=PLOTS_DIR):
    """Two normalised confusion matrices side by side."""
    os.makedirs(save_dir, exist_ok=True)
    cm_rf   = confusion_matrix(y_test, y_pred_rf)
    cm_lstm = confusion_matrix(y_test, y_pred_lstm)
    cm_rf_n   = cm_rf.astype(float)   / cm_rf.sum(axis=1, keepdims=True)
    cm_lstm_n = cm_lstm.astype(float) / cm_lstm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm, title, cmap in zip(
        axes,
        [cm_rf_n, cm_lstm_n],
        ["Random Forest (Normalised)", "LSTM (Normalised)"],
        ["Blues", "Purples"]
    ):
        sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=label_names, yticklabels=label_names,
                    linewidths=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Confusion Matrix Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Dual confusion matrix saved → {path}")


def print_comparison_table(rf_metrics, lstm_metrics):
    """Print a formatted comparison table to the console."""
    print("\n" + "="*52)
    print(f"{'Metric':<15} {'Random Forest':>15} {'LSTM':>15}")
    print("="*52)
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key.capitalize():<15} {rf_metrics[key]:>15.4f} {lstm_metrics[key]:>15.4f}")
    print("="*52)

    winner_f1 = "Random Forest" if rf_metrics["f1"] >= lstm_metrics["f1"] else "LSTM"
    print(f"\n🏆 Best F1-Score: {winner_f1}")


def run_comparison(X_test, y_test):
    """Load models and run full comparison pipeline."""
    rf_model, lstm_model, scaler, le, top_indices = load_models_and_data()
    y_pred_rf, y_pred_lstm = get_predictions(rf_model, lstm_model, X_test, y_test)

    rf_metrics   = compute_metrics(y_test, y_pred_rf)
    lstm_metrics = compute_metrics(y_test, y_pred_lstm)

    print_comparison_table(rf_metrics, lstm_metrics)

    print("\n🖼  Generating comparison plots...")
    plot_comparison_bar(rf_metrics, lstm_metrics)
    plot_dual_confusion_matrices(y_test, y_pred_rf, y_pred_lstm)

    return rf_metrics, lstm_metrics


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    data = run_preprocessing(
        train_path="data/KDDTrain+.txt",
        test_path="data/KDDTest+.txt"
    )
    run_comparison(data["X_test"], data["y_test"])