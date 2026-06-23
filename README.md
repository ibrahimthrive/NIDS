# 🛡️ Network Intrusion Detection System (NIDS)
### Design and Implementation Using Random Forest and Deep Learning
**Dataset:** NSL-KDD | **Models:** Random Forest + LSTM | **Deployment:** Streamlit

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
│       └── best_lstm.keras
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

## 📊 Expected Results (NSL-KDD Benchmarks)

| Model | Accuracy | F1-Score (weighted) |
|-------|----------|---------------------|
| Random Forest | ~99.2% | ~99.1% |
| LSTM | ~98.5% | ~98.4% |

> Results may vary slightly depending on random seeds and hardware.

---

## 🧠 Model Architecture

### Random Forest
- 200 trees, `max_features='sqrt'`, `class_weight='balanced'`
- Feature importance used for top-40 feature selection

### LSTM
```
Input (1, 40)
  → LSTM(128, return_sequences=True) → BatchNorm → Dropout(0.3)
  → LSTM(64)                         → BatchNorm → Dropout(0.3)
  → Dense(64, relu)                  → Dropout(0.15)
  → Dense(5, softmax)
```
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping (patience=7), ReduceLROnPlateau, ModelCheckpoint

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
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.4.0 | RF, metrics, preprocessing |
| tensorflow | 2.15.0 | LSTM deep learning |
| matplotlib | 3.8.2 | Evaluation plots |
| seaborn | 0.13.2 | Heatmaps |
| joblib | 1.3.2 | Model serialisation |
| streamlit | 1.31.0 | Web deployment |
| plotly | 5.18.0 | Interactive charts |
# NIDS
# NIDS
# NIDS
