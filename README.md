# 🛡️ Network Intrusion Detection System (NIDS)
### Design and Implementation Using Random Forest and Deep Learning
**Dataset:** NSL-KDD | **Models:** Random Forest + LSTM | **Deployment:** Streamlit

🌐 **Live demo:** [netids.streamlit.app](https://netids.streamlit.app)

---

## 📁 Project Structure

```
nids_project/
│
├── data/                        ← Place dataset files here
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── artifacts/                   ← Auto-created: saved models & preprocessing objects
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── selected_feature_indices.pkl
│   ├── selected_feature_names.pkl
│   ├── all_feature_names.pkl
│   ├── feature_importances.npy
│   ├── random_forest_model.pkl
│   └── lstm_model/
│       ├── lstm_final.keras
│       ├── best_val_acc.keras
│       └── best_minority_f1.keras
│
├── plots/                       ← Auto-created: evaluation plots
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importances.png
│   ├── rf_per_class_metrics.png
│   ├── lstm_confusion_matrix.png
│   ├── lstm_training_history.png
│   ├── lstm_per_class_metrics.png
│   ├── model_comparison.png
│   └── comparison_confusion_matrices.png
│
├── preprocessing.py             ← Data loading, encoding, scaling, feature selection
├── train_random_forest.py       ← RF model training and evaluation
├── train_lstm.py                ← LSTM model training and evaluation
├── compare_models.py            ← Side-by-side model comparison
├── predict.py                   ← Inference utility (NIDSPredictor class)
├── main.py                      ← Master pipeline orchestrator
├── app.py                       ← Streamlit web application
└── requirements.txt             ← Python dependencies
```

---

## ⚙️ Setup (VS Code)

### 1. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download **NSL-KDD** from https://www.unb.ca/cic/datasets/nsl.html  
Place the two files inside the `data/` folder:
```
data/KDDTrain+.txt
data/KDDTest+.txt
```

---

## 🚀 Running the Project

### Option A — Full Pipeline (Recommended)
Runs preprocessing → RF training → LSTM training → comparison in one command:
```bash
python main.py
```

Optional flags:
```bash
# Skip LSTM training (much faster, RF only)
python main.py --skip-lstm

# Custom dataset paths
python main.py --train path/to/KDDTrain+.txt --test path/to/KDDTest+.txt
```

### Option B — Run Each Step Individually
```bash
# Step 1: Preprocessing only
python preprocessing.py

# Step 2a: Train Random Forest
python train_random_forest.py

# Step 2b: Train LSTM
python train_lstm.py

# Step 3: Compare models
python compare_models.py
```

### Option C — CLI Inference on a Single Record
```bash
python predict.py
```

---

## 🌐 Launching the Web App

After training is complete, start the Streamlit app:
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** and provides:

| Tab | Description |
|-----|-------------|
| 🏠 Home | Overview, class distribution chart |
| 🔍 Live Predict | Interactive form to classify a single connection |
| 📁 Batch Predict | Upload CSV → predict all rows → download results |
| 📊 Model Metrics | View all evaluation plots and comparison charts |
| ℹ️ About | Dataset info, methodology, tech stack |

---

## 📊 Results (NSL-KDD Test Set)

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1-Score (weighted) |
|-------|----------|-----------------------|--------------------|----------------------|
| Random Forest | 75.6% | 78.9% | 75.6% | 70.8% |
| LSTM | 76.3% | 79.4% | 76.3% | 73.1% |

> Scores are on **KDDTest+**, which deliberately includes attack types absent
> from training (NSL-KDD's known difficulty). Both models still struggle on
> the rarest class, U2R (F1 well under 0.20) — see `plots/` for the full
> per-class breakdown. Results will vary with random seeds, hardware, and the
> RF search space in `train_random_forest.py`.

---

## 🧠 Model Architecture

### Random Forest
- `RandomizedSearchCV` over n_estimators, max_depth, max_features, class_weight, criterion (`train_random_forest.py`)
- Optimised for macro-F1 (not just weighted) so the rare classes matter; minority classes are oversampled before the search
- A separate lightweight Random Forest ranks feature importance for top-40 feature selection in `preprocessing.py`

### LSTM (BiLSTM v2 — Imbalance-Aware)
```
Input (1, 40)
  → BiLSTM(256, return_sequences=True) → LayerNorm → Dropout(0.4)
  → BiLSTM(128, return_sequences=True) → LayerNorm → Dropout(0.4)
  → LSTM(64)                           → LayerNorm → Dropout(0.4)
  → Dense(128, swish) → BatchNorm → Dropout(0.2)
  → Dense(64, swish)  → BatchNorm → Dropout(0.2)
  → Dense(5, softmax)
```
- Optimizer: Adam (lr=0.001, clipnorm=1.0)
- Loss: Focal Loss (gamma=2.5, alpha=0.25) with label smoothing (0.05)
- LR schedule: cosine annealing with warm restarts (T₀=20 epochs)
- Oversampling: SMOTETomek upsamples R2L → 8,000 and U2R → 2,000 samples before training
- Callbacks: EarlyStopping (patience=15), ModelCheckpoint (best val-accuracy), and a custom callback that saves the checkpoint with the best minority-class (R2L/U2R) F1

---

## 🏷️ Attack Class Mapping

| Class | Attack Types |
|-------|-------------|
| Normal | normal |
| DoS | back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, … |
| Probe | ipsweep, nmap, portsweep, satan, mscan, saint |
| R2L | ftp_write, guess_passwd, imap, multihop, phf, warezclient, … |
| U2R | buffer_overflow, loadmodule, perl, rootkit, sqlattack, … |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 3.0.3 | Data manipulation |
| numpy | 2.4.6 | Numerical computing |
| scikit-learn | 1.9.0 | RF, metrics, preprocessing |
| tensorflow | 2.21.0 | LSTM deep learning |
| matplotlib | 3.11.0 | Evaluation plots |
| seaborn | 0.13.2 | Heatmaps |
| joblib | 1.5.3 | Model serialisation |
| streamlit | 1.58.0 | Web deployment |
| plotly | 6.8.0 | Interactive charts |
| Pillow | 12.2.0 | Image handling in the Streamlit app |
| imbalanced-learn | 0.12.0 | SMOTE oversampling (LSTM pipeline) |
