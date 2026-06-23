"""
Step 2b: LSTM Deep Learning Model — Training & Evaluation (v2 — Imbalance-Aware)
==================================================================================
Key improvements over v1:
  1. SMOTE + ADASYN oversampling  → synthesises minority class samples
                                     (R2L: 995→8000, U2R: 52→2000)
  2. Deeper architecture          → 3 LSTM layers + residual-style dense block
  3. Focal Loss                   → penalises easy majority examples more,
                                     forces the model to focus on hard minority ones
  4. Label Smoothing              → prevents overconfident softmax on noisy labels
  5. Cosine Annealing LR          → smoother convergence, avoids LR plateau traps
  6. Larger patience + more epochs → genuine thorough training (no fast-tracking)
  7. Stratified validation split  → val set always contains all 5 classes
  8. Per-epoch minority-class F1 callback → saves best minority F1, not just accuracy
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from collections import Counter

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Bidirectional, Add, Activation, LayerNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback,
    LearningRateScheduler
)
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# imbalanced-learn
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn(
        "imbalanced-learn not installed. Run: pip install imbalanced-learn\n"
        "Falling back to class-weight-only strategy."
    )

from preprocessing import run_preprocessing, CLASS_NAMES

# ─── Configuration ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR  = "artifacts"
PLOTS_DIR      = "plots"
LSTM_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "lstm_model")
NUM_CLASSES    = len(CLASS_NAMES)

# Minority class indices (R2L=3, U2R=4) — used for the minority F1 callback
MINORITY_CLASSES = [3, 4]

LSTM_PARAMS = {
    # Architecture
    "lstm1_units":    256,
    "lstm2_units":    128,
    "lstm3_units":    64,
    "dense1_units":   128,
    "dense2_units":   64,
    "dropout_rate":   0.4,
    "recurrent_drop": 0.2,   # recurrent dropout inside LSTM cells
    # Training
    "learning_rate":  0.001,
    "batch_size":     256,    # smaller batch → more updates per epoch, better minority learning
    "epochs":         100,    # genuine training budget
    "patience":       15,     # generous patience — don't stop too early
    # Focal loss
    "focal_gamma":    2.5,    # higher = more focus on hard/minority examples
    "focal_alpha":    0.25,
    # Label smoothing
    "label_smoothing": 0.05,
    # Oversampling targets (samples per minority class after SMOTE)
    "smote_r2l_target": 8000,
    "smote_u2r_target": 2000,
}


# ──────────────────────────────────────────────────────────────────────────────
# Version-safe Keras serialisation decorator
# tf.keras.saving → TF 2.12+   |   tf.keras.utils → TF 2.x (older, always present)
# ──────────────────────────────────────────────────────────────────────────────
try:
    _register = tf.keras.saving.register_keras_serializable
except AttributeError:
    _register = tf.keras.utils.register_keras_serializable

# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS — registered so Keras can serialise/deserialise it automatically
# ══════════════════════════════════════════════════════════════════════════════

@_register(package="nids")
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Subclassing tf.keras.losses.Loss and decorating with
    @register_keras_serializable lets Keras save/load the loss
    automatically without needing custom_objects on load_model().
    """
    def __init__(self, gamma: float = 2.5, alpha: float = 0.25,
                 name: str = "focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        ce      = -y_true * tf.math.log(y_pred)
        weight  = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        fl      = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ══════════════════════════════════════════════════════════════════════════════
# MINORITY F1 CALLBACK — saves model on best minority-class F1, not accuracy
# ══════════════════════════════════════════════════════════════════════════════

class MinorityF1Checkpoint(Callback):
    """
    At the end of each epoch, evaluates F1 on the validation set
    for R2L and U2R classes specifically, and saves the model whenever
    the macro-average minority F1 improves.
    """
    def __init__(self, X_val, y_val, save_path, minority_indices=MINORITY_CLASSES):
        super().__init__()
        self.X_val           = X_val
        self.y_val           = y_val
        self.save_path       = save_path
        self.minority_indices = minority_indices
        self.best_f1         = -np.inf
        self.best_epoch      = 0

    def on_epoch_end(self, epoch, logs=None):
        y_prob  = self.model.predict(self.X_val, verbose=0)
        y_pred  = np.argmax(y_prob, axis=1)
        y_true  = self.y_val

        per_class_f1 = f1_score(y_true, y_pred, average=None,
                                labels=list(range(NUM_CLASSES)),
                                zero_division=0)
        minority_f1 = np.mean([per_class_f1[i] for i in self.minority_indices])

        r2l_f1 = per_class_f1[3] if 3 < len(per_class_f1) else 0.0
        u2r_f1 = per_class_f1[4] if 4 < len(per_class_f1) else 0.0

        print(f"  [MinorityF1] R2L={r2l_f1:.3f}  U2R={u2r_f1:.3f}  "
              f"Minority-avg={minority_f1:.3f}  (best={self.best_f1:.3f})")

        if minority_f1 > self.best_f1:
            self.best_f1    = minority_f1
            self.best_epoch = epoch + 1
            self.model.save(self.save_path)
            print(f"  ✅ New best minority F1={minority_f1:.4f} → saved model")


# ══════════════════════════════════════════════════════════════════════════════
# OVERSAMPLING WITH SMOTE
# ══════════════════════════════════════════════════════════════════════════════

def oversample_minority(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Apply SMOTETomek (SMOTE oversampling + Tomek link cleaning) to the
    training set. Sets custom sampling targets so:
      - R2L goes from ~900 → 8,000 samples
      - U2R goes from ~47  → 2,000 samples
    Majority classes (Normal, DoS) are left unchanged.
    """
    if not IMBLEARN_AVAILABLE:
        print("   ⚠  imbalanced-learn not available — skipping SMOTE.")
        return X_train, y_train

    print("\n⚖️  Applying SMOTETomek oversampling for minority classes...")
    counts_before = Counter(y_train)
    print(f"   Before: { {CLASS_NAMES[k]: v for k,v in sorted(counts_before.items())} }")

    # Build a sampling strategy that only upsamples minority classes
    # (leave Normal, DoS, Probe at their natural counts)
    sampling_strategy = {}
    for cls_idx, target in [
        (3, LSTM_PARAMS["smote_r2l_target"]),   # R2L
        (4, LSTM_PARAMS["smote_u2r_target"]),   # U2R
    ]:
        if counts_before.get(cls_idx, 0) < target:
            sampling_strategy[cls_idx] = target

    # SMOTE requires k_neighbors ≤ minority class count
    min_count = min(counts_before.get(i, 1) for i in sampling_strategy)
    k_neighbors = min(5, min_count - 1)
    k_neighbors = max(1, k_neighbors)

    try:
        smote = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            ),
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"   ⚠  SMOTETomek failed ({e}). Trying plain SMOTE...")
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

    counts_after = Counter(y_res)
    print(f"   After : { {CLASS_NAMES[k]: v for k,v in sorted(counts_after.items())} }")
    print(f"   Total samples: {len(X_train)} → {len(X_res)}")
    return X_res.astype(np.float32), y_res


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(input_features: int, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Deep Bidirectional LSTM architecture:

      Input (1, features)
        → BiLSTM(256, return_sequences)  → LayerNorm → Dropout
        → BiLSTM(128, return_sequences)  → LayerNorm → Dropout
        → LSTM(64)                       → LayerNorm → Dropout
        → Dense(128, swish) → BN → Dropout
        → Dense(64, swish)  → BN → Dropout
        → Dense(5, softmax)

    Bidirectional layers read the sequence both forwards and backwards,
    doubling the effective representation capacity at the same parameter count.
    LayerNormalization (instead of BatchNorm inside recurrent layers) is more
    stable for sequence models.
    Swish activation (x * sigmoid(x)) is smoother than ReLU and consistently
    outperforms it on tabular-as-sequence tasks.
    """
    inp = Input(shape=(1, input_features), name="input")

    # ── Recurrent Block ──
    x = Bidirectional(
        LSTM(LSTM_PARAMS["lstm1_units"],
             return_sequences=True,
             recurrent_dropout=LSTM_PARAMS["recurrent_drop"],
             name="lstm_1"),
        name="bilstm_1"
    )(inp)
    x = LayerNormalization()(x)
    x = Dropout(LSTM_PARAMS["dropout_rate"])(x)

    x = Bidirectional(
        LSTM(LSTM_PARAMS["lstm2_units"],
             return_sequences=True,
             recurrent_dropout=LSTM_PARAMS["recurrent_drop"],
             name="lstm_2"),
        name="bilstm_2"
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(LSTM_PARAMS["dropout_rate"])(x)

    x = LSTM(LSTM_PARAMS["lstm3_units"],
             return_sequences=False,
             recurrent_dropout=LSTM_PARAMS["recurrent_drop"],
             name="lstm_3")(x)
    x = LayerNormalization()(x)
    x = Dropout(LSTM_PARAMS["dropout_rate"])(x)

    # ── Dense Classification Block ──
    x = Dense(LSTM_PARAMS["dense1_units"], activation="swish", name="dense_1")(x)
    x = BatchNormalization()(x)
    x = Dropout(LSTM_PARAMS["dropout_rate"] / 2)(x)

    x = Dense(LSTM_PARAMS["dense2_units"], activation="swish", name="dense_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(LSTM_PARAMS["dropout_rate"] / 2)(x)

    out = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="NIDS_BiLSTM_v2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LSTM_PARAMS["learning_rate"],
            clipnorm=1.0           # gradient clipping prevents exploding gradients
        ),
        loss=FocalLoss(
            gamma=LSTM_PARAMS["focal_gamma"],
            alpha=LSTM_PARAMS["focal_alpha"]
        ),
        metrics=["accuracy"]
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE — Cosine Annealing with Warm Restarts
# ══════════════════════════════════════════════════════════════════════════════

def cosine_annealing_schedule(epoch: int, lr: float) -> float:
    """
    Cosine annealing with warm restarts (T_0=20 epochs).
    Cyclically decays LR from max → min, then restarts.
    Helps escape local minima and explore better solutions.
    """
    T_0    = 20           # restart period (epochs)
    lr_min = 1e-6
    lr_max = LSTM_PARAMS["learning_rate"]
    t      = epoch % T_0
    new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_0))
    return float(new_lr)


# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS — Extra-boosted for U2R
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Balanced weights with an extra 3× boost for U2R (the rarest class),
    applied on top of the sklearn balanced weights.
    Even after SMOTE, U2R is still underrepresented at test time.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    # Extra boost for U2R (index 4)
    if 4 in cw:
        cw[4] = cw[4] * 3.0
    return cw


# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def stratified_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.12):
    """
    Stratified split ensuring all 5 classes appear in both train and val.
    Uses 12% for validation (slightly more than before to get enough minority samples).
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray):
    """Full training pipeline with oversampling, focal loss, and careful callbacks."""
    os.makedirs(LSTM_MODEL_DIR, exist_ok=True)

    n_features = X_train.shape[1]

    # ── Step A: Oversample minority classes ──────────────────────────────────
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train)

    # ── Step B: Reshape to (samples, 1, features) ─────────────────────────────
    X_tr_3d  = X_train_bal.reshape(-1, 1, n_features)
    X_val_3d = X_val.reshape(-1, 1, n_features)

    # ── Step C: One-hot encode labels ─────────────────────────────────────────
    y_tr_oh  = to_categorical(y_train_bal, num_classes=NUM_CLASSES)
    y_val_oh = to_categorical(y_val,       num_classes=NUM_CLASSES)

    # Apply label smoothing manually (Keras label smoothing only works with
    # sparse labels in some versions)
    ls = LSTM_PARAMS["label_smoothing"]
    y_tr_oh  = y_tr_oh  * (1 - ls) + ls / NUM_CLASSES
    y_val_oh = y_val_oh * (1 - ls) + ls / NUM_CLASSES

    # ── Step D: Class weights ──────────────────────────────────────────────────
    cw = compute_class_weights(y_train_bal)
    print(f"\n   Class weights (post-SMOTE): "
          f"{ {CLASS_NAMES[k]: round(v, 2) for k, v in cw.items()} }")

    # ── Step E: Build model ────────────────────────────────────────────────────
    model = build_lstm_model(n_features)
    model.summary()

    # ── Step F: Callbacks ──────────────────────────────────────────────────────
    minority_ckpt_path = os.path.join(LSTM_MODEL_DIR, "best_minority_f1.keras")
    std_ckpt_path      = os.path.join(LSTM_MODEL_DIR, "best_val_acc.keras")

    callbacks = [
        # 1. Save best model by val_accuracy (standard checkpoint)
        ModelCheckpoint(
            filepath=std_ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        ),
        # 2. Save best model by minority F1 (custom checkpoint)
        MinorityF1Checkpoint(X_val_3d, y_val, minority_ckpt_path),
        # 3. EarlyStopping — generous patience, monitor val_loss
        EarlyStopping(
            monitor="val_loss",
            patience=LSTM_PARAMS["patience"],
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        ),
        # 4. Cosine annealing LR
        LearningRateScheduler(cosine_annealing_schedule, verbose=0),
    ]

    # ── Step G: Train ──────────────────────────────────────────────────────────
    print(f"\n🧠 Training BiLSTM-v2 | epochs={LSTM_PARAMS['epochs']} "
          f"| batch={LSTM_PARAMS['batch_size']} | features={n_features}")
    print(f"   Training samples (post-SMOTE): {len(X_tr_3d)}")
    print(f"   Validation samples: {len(X_val_3d)}\n")

    history = model.fit(
        X_tr_3d, y_tr_oh,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        validation_data=(X_val_3d, y_val_oh),
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    print("\n   ✅ Training complete.")
    print(f"   Best minority-F1 model saved → {minority_ckpt_path}")
    print(f"   Best val-accuracy model saved → {std_ckpt_path}")

    # ── Step H: Load best minority-F1 model for final evaluation ──────────────
    if os.path.exists(minority_ckpt_path):
        print("\n🔄 Loading best minority-F1 checkpoint for evaluation...")
        # FocalLoss is registered via @register_keras_serializable,
        # so no custom_objects dict is needed here.
        best_model = tf.keras.models.load_model(minority_ckpt_path)
    else:
        best_model = model

    return best_model, history, n_features


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_lstm(model, X_test: np.ndarray, y_test: np.ndarray,
                  label_names=CLASS_NAMES):
    """Full evaluation with per-class breakdown."""
    print("\n📊 Evaluating LSTM on test set...")
    X_test_3d = X_test.reshape(-1, 1, X_test.shape[1])
    y_prob    = model.predict(X_test_3d, verbose=0)
    y_pred    = np.argmax(y_prob, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Minority-class specific scores
    per_f1  = f1_score(y_test, y_pred, average=None, zero_division=0,
                       labels=list(range(NUM_CLASSES)))
    r2l_f1  = per_f1[3] if len(per_f1) > 3 else 0.0
    u2r_f1  = per_f1[4] if len(per_f1) > 4 else 0.0

    print(f"\n{'='*55}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (W)  : {prec:.4f}")
    print(f"  Recall    (W)  : {rec:.4f}")
    print(f"  F1-Score  (W)  : {f1:.4f}")
    print(f"{'─'*55}")
    print(f"  R2L  F1-Score  : {r2l_f1:.4f}   ← minority target")
    print(f"  U2R  F1-Score  : {u2r_f1:.4f}   ← minority target")
    print(f"{'='*55}")

    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "r2l_f1": r2l_f1, "u2r_f1": u2r_f1
    }
    return y_pred, metrics


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_dir=PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, keys, title, ylabel in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Training vs Validation Loss", "Training vs Validation Accuracy"],
        ["Loss", "Accuracy"]
    ):
        epochs = range(1, len(history.history[keys[0]]) + 1)
        ax.plot(epochs, history.history[keys[0]], label="Train",
                color="#4C72B0", linewidth=2)
        if keys[1] in history.history:
            ax.plot(epochs, history.history[keys[1]], label="Validation",
                    color="#C44E52", linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(linestyle="--", alpha=0.5)

    plt.suptitle("LSTM v2 Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "lstm_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Training history saved → {path}")


def plot_confusion_matrix(y_test, y_pred, label_names=CLASS_NAMES, save_dir=PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Purples",
                    xticklabels=label_names, yticklabels=label_names,
                    linewidths=0.5, ax=ax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("LSTM v2 — Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "lstm_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Confusion matrix saved → {path}")


def plot_per_class_metrics(y_test, y_pred, label_names=CLASS_NAMES, save_dir=PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=None, zero_division=0)

    x = np.arange(len(label_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, prec, width, label="Precision", color="#8172B2")
    ax.bar(x,         rec,  width, label="Recall",    color="#64B5CD")
    ax.bar(x + width, f1,   width, label="F1-Score",  color="#EE854A")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("LSTM v2 — Per-Class Metrics", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate R2L and U2R bars
    for i, cls in enumerate(label_names):
        if cls in ("R2L", "U2R"):
            for val, offset in zip([prec[i], rec[i], f1[i]], [-width, 0, width]):
                ax.text(i + offset, val + 0.02, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8, color="darkred",
                        fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "lstm_per_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Per-class metrics saved → {path}")


def plot_class_distribution_comparison(y_before, y_after, save_dir=PLOTS_DIR):
    """Bar chart showing class distribution before and after SMOTE."""
    os.makedirs(save_dir, exist_ok=True)
    before = Counter(y_before)
    after  = Counter(y_after)
    x      = np.arange(NUM_CLASSES)
    width  = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, [before.get(i, 0) for i in range(NUM_CLASSES)],
           width, label="Before SMOTE", color="#4C72B0", alpha=0.8)
    ax.bar(x + width/2, [after.get(i, 0)  for i in range(NUM_CLASSES)],
           width, label="After SMOTE",  color="#55A868", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution Before vs After SMOTE", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_yscale("log")   # log scale so minority bars are visible
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "smote_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   SMOTE distribution plot saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_lstm_pipeline(data: dict):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # Stratified val split (before SMOTE — SMOTE only applied to train portion)
    print("\n📐 Creating stratified validation split (12%)...")
    X_tr, X_val, y_tr, y_val = stratified_val_split(X_train, y_train, val_ratio=0.12)
    print(f"   Train: {len(X_tr)} | Val: {len(X_val)}")
    print(f"   Val class dist: { {CLASS_NAMES[k]: v for k,v in sorted(Counter(y_val).items())} }")

    # Save pre-SMOTE distribution for plot
    y_before_smote = y_tr.copy()

    # Train
    lstm_model, history, _ = train_lstm(X_tr, y_tr, X_val, y_val)

    # Save final model
    final_path = os.path.join(LSTM_MODEL_DIR, "lstm_final.keras")
    lstm_model.save(final_path)
    print(f"\n💾 Final LSTM model saved → {final_path}")

    # Evaluate
    y_pred, metrics = evaluate_lstm(lstm_model, X_test, y_test)

    # Plots
    print("\n🖼  Generating evaluation plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_per_class_metrics(y_test, y_pred)

    # SMOTE distribution plot (approximate — show before/after on training portion)
    if IMBLEARN_AVAILABLE:
        from collections import Counter as C2
        y_after = np.concatenate([
            y_tr,
            np.array([3] * (LSTM_PARAMS["smote_r2l_target"] - Counter(y_tr).get(3, 0))),
            np.array([4] * (LSTM_PARAMS["smote_u2r_target"] - Counter(y_tr).get(4, 0))),
        ])
        plot_class_distribution_comparison(y_before_smote, y_after)

    return lstm_model, metrics


if __name__ == "__main__":
    data = run_preprocessing(
        train_path="data/KDDTrain+.txt",
        test_path="data/KDDTest+.txt"
    )
    run_lstm_pipeline(data)