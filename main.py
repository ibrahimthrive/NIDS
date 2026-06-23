"""
Step 4: Main Training Pipeline
================================
Orchestrates the full pipeline:
  preprocessing → RF training → LSTM training → comparison
Run this single script to train and evaluate both models end-to-end.
"""

import os
import time
import argparse
from preprocessing      import run_preprocessing
from train_random_forest import run_rf_pipeline
from train_lstm          import run_lstm_pipeline
from compare_models      import run_comparison


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Network Intrusion Detection System (NIDS)                  ║
║   Random Forest + LSTM  |  NSL-KDD Dataset                   ║
╚══════════════════════════════════════════════════════════════╝
""")


def main(train_path: str, test_path: str, skip_lstm: bool = False):
    print_banner()
    total_start = time.time()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("STEP 1 / 4  ▶  Data Preprocessing")
    print("─"*60)
    t0 = time.time()
    data = run_preprocessing(train_path, test_path)
    print(f"⏱  Done in {time.time()-t0:.1f}s\n")

    # ── Step 2: Random Forest ─────────────────────────────────────────────────
    print("─"*60)
    print("STEP 2 / 4  ▶  Random Forest Training & Evaluation")
    print("─"*60)
    t0 = time.time()
    rf_model, rf_metrics = run_rf_pipeline(data)
    print(f"⏱  Done in {time.time()-t0:.1f}s\n")

    # ── Step 3: LSTM ──────────────────────────────────────────────────────────
    lstm_metrics = None
    if not skip_lstm:
        print("─"*60)
        print("STEP 3 / 4  ▶  LSTM Training & Evaluation")
        print("─"*60)
        t0 = time.time()
        lstm_model, lstm_metrics = run_lstm_pipeline(data)
        print(f"⏱  Done in {time.time()-t0:.1f}s\n")

    # ── Step 4: Comparison ────────────────────────────────────────────────────
    if lstm_metrics is not None:
        print("─"*60)
        print("STEP 4 / 4  ▶  Model Comparison")
        print("─"*60)
        t0 = time.time()
        run_comparison(data["X_test"], data["y_test"])
        print(f"⏱  Done in {time.time()-t0:.1f}s\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "═"*60)
    print(f"✅  Full pipeline complete in {elapsed/60:.1f} min")
    print("   Artifacts → artifacts/")
    print("   Plots     → plots/")
    print("   Run the app: streamlit run app.py")
    print("═"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIDS Training Pipeline")
    parser.add_argument("--train", default="data/KDDTrain+.txt", help="Path to KDDTrain+.txt")
    parser.add_argument("--test",  default="data/KDDTest+.txt",  help="Path to KDDTest+.txt")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (faster, RF only)")
    args = parser.parse_args()
    main(args.train, args.test, skip_lstm=args.skip_lstm)
