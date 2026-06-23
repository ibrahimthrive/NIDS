"""
Step 1: Data Preprocessing
===========================
Loads NSL-KDD dataset, maps attack labels to 5 classes, handles missing values,
one-hot encodes categorical features, aligns train/test columns, scales features,
and performs Random Forest-based feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ─── Column Definitions ────────────────────────────────────────────────────────
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# ─── Attack-to-Class Mapping ───────────────────────────────────────────────────
ATTACK_MAP = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
NUM_FEATURES_TO_SELECT = 40  # RF-based feature selection target


def load_dataset(train_path: str, test_path: str):
    """Load NSL-KDD train and test files."""
    print("📂 Loading datasets...")
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=COLUMNS)

    # Drop the difficulty level column (not a feature)
    train_df.drop(columns=["difficulty_level"], inplace=True)
    test_df.drop(columns=["difficulty_level"],  inplace=True)

    print(f"   Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    return train_df, test_df


def map_attack_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw attack strings to 5-class labels."""
    df = df.copy()
    df["label"] = df["label"].str.strip().str.lower().map(ATTACK_MAP)
    unknown = df["label"].isna().sum()
    if unknown > 0:
        print(f"   ⚠  {unknown} unknown labels set to 'Normal'")
        df["label"].fillna("Normal", inplace=True)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaN with median, categorical NaN with mode.
    Compatible with pandas 2.x Copy-on-Write semantics."""
    df = df.copy()
    for col in df.columns:
        if col == "label":
            continue
        # Use pandas api_types to reliably detect numeric vs string/object
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_vals = df[col].mode()
            fill_val  = mode_vals[0] if len(mode_vals) > 0 else "unknown"
            df[col]   = df[col].fillna(fill_val)
    return df


def one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode categorical columns and align train/test."""
    print("🔢 One-hot encoding categorical features...")
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLS)
    test_df  = pd.get_dummies(test_df,  columns=CATEGORICAL_COLS)

    # Align columns: add missing cols as 0, drop extra cols
    train_cols = set(train_df.columns)
    test_cols  = set(test_df.columns)

    for col in train_cols - test_cols:
        test_df[col] = 0
    for col in test_cols - train_cols:
        train_df[col] = 0

    # Ensure identical column order (label last)
    feature_cols = sorted([c for c in train_df.columns if c != "label"])
    train_df = train_df[feature_cols + ["label"]]
    test_df  = test_df[feature_cols  + ["label"]]

    print(f"   Features after encoding: {len(feature_cols)}")
    return train_df, test_df


def encode_labels(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Encode string class labels to integers."""
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["label"] = le.transform(train_df["label"])
    test_df["label"]  = le.transform(test_df["label"])
    return train_df, test_df, le


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Fit StandardScaler on train, transform both splits."""
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def rf_feature_selection(X_train: np.ndarray, y_train: np.ndarray,
                          feature_names: list, n_features: int = NUM_FEATURES_TO_SELECT):
    """
    Train a lightweight Random Forest to rank features by importance,
    then return the indices of the top-n features.
    """
    print(f"🌲 Running RF-based feature selection (top {n_features} features)...")
    selector_rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
    )
    selector_rf.fit(X_train, y_train)

    importances = selector_rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:n_features]
    selected_features = [feature_names[i] for i in top_indices]

    print(f"   Selected features: {selected_features[:10]} ... (showing top 10)")
    return top_indices, selected_features, importances


def run_preprocessing(train_path: str, test_path: str, artifacts_dir: str = "artifacts"):
    """Full preprocessing pipeline — returns ready-to-use arrays and metadata."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load
    train_df, test_df = load_dataset(train_path, test_path)

    # 2. Map labels
    print("🏷  Mapping attack labels...")
    train_df = map_attack_labels(train_df)
    test_df  = map_attack_labels(test_df)
    print("   Class distribution (train):")
    print("  ", train_df["label"].value_counts().to_dict())

    # 3. Handle missing values
    train_df = handle_missing_values(train_df)
    test_df  = handle_missing_values(test_df)

    # 4. One-hot encode + align
    train_df, test_df = one_hot_encode(train_df, test_df)

    # 5. Encode labels
    train_df, test_df, label_encoder = encode_labels(train_df, test_df)

    # 6. Split features / labels
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values

    # 7. Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 8. RF feature selection
    top_indices, selected_features, importances = rf_feature_selection(
        X_train_scaled, y_train, feature_cols
    )
    X_train_sel = X_train_scaled[:, top_indices]
    X_test_sel  = X_test_scaled[:, top_indices]

    print(f"\n✅ Preprocessing complete.")
    print(f"   X_train: {X_train_sel.shape} | X_test: {X_test_sel.shape}")

    # Save artifacts
    joblib.dump(scaler,         os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(label_encoder,  os.path.join(artifacts_dir, "label_encoder.pkl"))
    joblib.dump(top_indices,    os.path.join(artifacts_dir, "selected_feature_indices.pkl"))
    joblib.dump(selected_features, os.path.join(artifacts_dir, "selected_feature_names.pkl"))
    joblib.dump(feature_cols,   os.path.join(artifacts_dir, "all_feature_names.pkl"))
    np.save(os.path.join(artifacts_dir, "feature_importances.npy"), importances)
    print(f"   Artifacts saved to '{artifacts_dir}/'")

    return {
        "X_train": X_train_sel,
        "X_test":  X_test_sel,
        "y_train": y_train,
        "y_test":  y_test,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "selected_features": selected_features,
        "top_indices": top_indices,
        "feature_importances": importances,
        "all_feature_names": feature_cols,
    }


if __name__ == "__main__":
    data = run_preprocessing(
        train_path="data/KDDTrain+.txt",
        test_path="data/KDDTest+.txt"
    )