"""
predict.py — Inference Utility
================================
Load trained artifacts and run predictions on new/unseen network traffic records.
Handles three upload formats automatically:
  1. Raw NSL-KDD .txt file (no header, 42 or 43 columns including label/difficulty)
  2. Raw CSV with named NSL-KDD headers
  3. Already one-hot-encoded CSV (e.g. exported from a previous run)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from preprocessing import COLUMNS, CATEGORICAL_COLS, ATTACK_MAP, CLASS_NAMES
# Register FocalLoss before any load_model call
from train_lstm import FocalLoss  # noqa: F401

# Windows consoles default to a non-UTF-8 codepage (e.g. cp1252), which raises
# UnicodeEncodeError on the emoji in the print() calls below and silently
# breaks model loading. Force UTF-8 stdout/stderr so this never crashes.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ARTIFACTS_DIR  = "artifacts"
RF_MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "random_forest_model.pkl")
LSTM_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "lstm_model", "lstm_final.keras")

# The 41 raw feature column names (no label, no difficulty_level)
RAW_FEATURE_COLS = [c for c in COLUMNS if c not in ("label", "difficulty_level")]


class NIDSPredictor:
    """
    Wraps preprocessing + model inference into a single reusable class.
    Supports both Random Forest and LSTM predictions.
    """

    def __init__(self, model_type: str = "rf"):
        assert model_type in ("rf", "lstm", "both"), \
            "model_type must be 'rf', 'lstm', or 'both'"
        self.model_type = model_type

        print(f"🔄 Loading artifacts (model={model_type})...")
        self.scaler         = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
        self.label_encoder  = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
        self.top_indices    = joblib.load(os.path.join(ARTIFACTS_DIR, "selected_feature_indices.pkl"))
        self.all_feat_names = joblib.load(os.path.join(ARTIFACTS_DIR, "all_feature_names.pkl"))

        self.rf_model   = None
        self.lstm_model = None

        if model_type in ("rf", "both"):
            self.rf_model = joblib.load(RF_MODEL_PATH)
            print("   ✅ Random Forest loaded.")

        if model_type in ("lstm", "both"):
            self.lstm_model = tf.keras.models.load_model(LSTM_MODEL_DIR)
            print("   ✅ LSTM loaded.")

    # ── Format detection ───────────────────────────────────────────────────────

    def _detect_and_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect the uploaded CSV format and normalise it to a raw-featured
        DataFrame with proper column names before one-hot encoding.

        Three cases handled:
          A) Raw NSL-KDD without headers (numeric column names 0..41 or 0..42)
             → assign standard column names, drop label/difficulty if present
          B) Raw NSL-KDD with named headers (protocol_type / service / flag present)
             → drop label/difficulty if present, ready for encoding
          C) Already one-hot-encoded (none of the raw categorical cols present,
             but training feature names are present)
             → skip encoding entirely, go straight to alignment + scale
        """
        df = df.copy()

        # ── Drop label / difficulty columns if they snuck in ──────────────────
        drop_candidates = ["label", "difficulty_level", "difficulty"]
        df.drop(columns=[c for c in drop_candidates if c in df.columns],
                inplace=True, errors="ignore")

        col0 = str(df.columns[0])

        # ── Case A: numeric column headers → raw NSL-KDD without header row ──
        if col0.lstrip("-").isdigit() or col0 == "0":
            n_cols = df.shape[1]
            if n_cols == 41:
                df.columns = RAW_FEATURE_COLS
            elif n_cols >= 42:
                # Extra columns at end (label and/or difficulty from txt file)
                df = df.iloc[:, :41]
                df.columns = RAW_FEATURE_COLS
            else:
                raise ValueError(
                    f"Expected 41 feature columns for raw NSL-KDD data, got {n_cols}."
                )
            return self._encode_raw(df)

        # ── Case B: named headers, categorical cols still string ──────────────
        if any(c in df.columns for c in CATEGORICAL_COLS):
            return self._encode_raw(df)

        # ── Case C: already encoded (training feature names present) ──────────
        # Just align columns and return
        return df

    def _encode_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the three categorical columns."""
        # Ensure categorical columns are string type
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
        return df

    # ── Core preprocessing ─────────────────────────────────────────────────────

    def _preprocess_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Full preprocessing pipeline:
          detect format → normalise → align columns → scale → feature-select
        """
        df = self._detect_and_normalise(df)

        # Align with the exact training feature set
        for col in self.all_feat_names:
            if col not in df.columns:
                df[col] = 0
        # Drop any extra columns not seen during training
        df = df[self.all_feat_names]

        # Scale
        X = df.values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        # Feature selection
        X_sel = X_scaled[:, self.top_indices]
        return X_sel

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict on a DataFrame of NSL-KDD records (any supported format).

        Returns dict with keys:
          rf_predictions / lstm_predictions / ensemble  → list of class name strings
          rf_probabilities / lstm_probabilities          → numpy probability arrays
        """
        X = self._preprocess_dataframe(df)
        results = {}

        # label_encoder.classes_ gives the canonical class order used for all
        # probability arrays throughout this predictor.
        enc_classes = list(self.label_encoder.classes_)

        if self.rf_model is not None:
            # RF predict_proba columns follow rf_model.classes_ order.
            # Reorder to match label_encoder.classes_ so both models align.
            rf_col_order = list(self.rf_model.classes_)
            rf_int_raw   = self.rf_model.predict(X)
            rf_prob_raw  = self.rf_model.predict_proba(X)

            # Build reorder index: for each encoder class, find its RF column
            rf_reorder = [rf_col_order.index(c) for c in enc_classes if c in rf_col_order]
            # If RF has all classes, reorder; otherwise use as-is
            if len(rf_reorder) == len(enc_classes):
                rf_prob = rf_prob_raw[:, rf_reorder]
            else:
                rf_prob = rf_prob_raw

            results["rf_predictions"]   = self.label_encoder.inverse_transform(rf_int_raw).tolist()
            results["rf_probabilities"] = rf_prob

        if self.lstm_model is not None:
            X_3d      = X.reshape(-1, 1, X.shape[1])
            lstm_prob = self.lstm_model.predict(X_3d, verbose=0)
            # LSTM softmax outputs are in label_encoder.classes_ order (same order
            # as to_categorical was applied during training)
            lstm_int  = np.argmax(lstm_prob, axis=1)
            results["lstm_predictions"]   = self.label_encoder.inverse_transform(lstm_int).tolist()
            results["lstm_probabilities"] = lstm_prob

        if self.rf_model is not None and self.lstm_model is not None:
            # Both arrays now in enc_classes order — safe to average
            avg_prob     = (results["rf_probabilities"] + results["lstm_probabilities"]) / 2
            ensemble_int = np.argmax(avg_prob, axis=1)
            results["ensemble"] = self.label_encoder.inverse_transform(ensemble_int).tolist()

        return results

    def predict_single(self, record: dict) -> dict:
        """
        Predict on a single traffic record dict (41 raw feature keys).
        Returns prediction label, confidence, per-class probabilities.
        """
        # Fill any missing keys with safe defaults
        feature_defaults = {col: 0 for col in RAW_FEATURE_COLS}
        feature_defaults["protocol_type"] = "tcp"
        feature_defaults["service"]       = "http"
        feature_defaults["flag"]          = "SF"
        feature_defaults.update(record)

        df  = pd.DataFrame([feature_defaults])
        raw = self.predict(df)

        if self.model_type == "rf":
            label = raw["rf_predictions"][0]
            probs = raw["rf_probabilities"][0]
        elif self.model_type == "lstm":
            label = raw["lstm_predictions"][0]
            probs = raw["lstm_probabilities"][0]
        else:
            label = raw["ensemble"][0]
            probs = (raw["rf_probabilities"][0] + raw["lstm_probabilities"][0]) / 2

        # CRITICAL: use label_encoder.classes_ (NOT CLASS_NAMES) to map probability
        # indices to class names. The encoder sorts classes alphabetically:
        # index 0=DoS, 1=Normal, 2=Probe, 3=R2L, 4=U2R
        # Using CLASS_NAMES directly would swap the labels and cause the chart
        # to show DoS probability under the "Normal" bar and vice-versa.
        enc_classes = self.label_encoder.classes_   # e.g. ["DoS","Normal","Probe","R2L","U2R"]

        confidence  = float(np.max(probs))
        class_probs = {enc_classes[i]: float(probs[i]) for i in range(len(enc_classes))}

        # Second-highest class and its probability (for uncertainty warning)
        sorted_probs  = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        top_class     = sorted_probs[0][0]
        runner_up     = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)
        is_uncertain  = confidence < 0.70 or runner_up[1] > 0.25

        # Recompute label from argmax of probs to guarantee it matches the chart
        # (do NOT trust the earlier label variable — it may have been decoded with
        #  a stale index before the probability array was reordered)
        label = top_class

        # Per-model votes (for ensemble transparency)
        rf_vote   = raw.get("rf_predictions",   [None])[0]
        lstm_vote = raw.get("lstm_predictions", [None])[0]
        rf_probs_dict   = (
            {enc_classes[i]: float(raw["rf_probabilities"][0][i])
             for i in range(len(enc_classes))}
            if "rf_probabilities" in raw else {}
        )
        lstm_probs_dict = (
            {enc_classes[i]: float(raw["lstm_probabilities"][0][i])
             for i in range(len(enc_classes))}
            if "lstm_probabilities" in raw else {}
        )

        return {
            "prediction":      label,
            "confidence":      confidence,
            "class_probs":     class_probs,
            "is_attack":       label != "Normal",
            "is_uncertain":    is_uncertain,
            "runner_up":       {"class": runner_up[0], "prob": runner_up[1]},
            "rf_vote":         rf_vote,
            "lstm_vote":       lstm_vote,
            "rf_probs":        rf_probs_dict,
            "lstm_probs":      lstm_probs_dict,
        }


# ─── CLI Usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    predictor = NIDSPredictor(model_type="both")

    sample = {
        "duration": 0, "protocol_type": "tcp", "service": "http",
        "flag": "SF", "src_bytes": 215, "dst_bytes": 45076,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 1,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 15,
        "dst_host_srv_count": 15, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.07,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }

    result = predictor.predict_single(sample)
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
