"""
Phase 2 Execution Script: Baseline Model Training & Single-Attribute Fairness Audit.

This script executes the complete Phase 2 workflow:
1. Loads raw Adult Census Income dataset.
2. Preprocesses data into decoupled features (X), target (y), and protected attributes (A).
3. Performs reproducible train-test split (80% train, 20% test).
4. Fits a baseline Logistic Regression classifier on training set X_train.
5. Evaluates model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix) on test set X_test.
6. Conducts single-attribute fairness audits for protected attributes ('sex', 'race').
7. Serializes baseline metrics and fairness audit results to 'data/processed/baseline_results.json'.
8. Prints clean metric summaries to stdout.
"""

import json
import os
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Add project root directory to path for clean execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_raw_data, load_config
from src.data.preprocessor import prepare_pipeline_data, split_pipeline_data
from src.models.classifier import train_baseline_model, predict_model, evaluate_performance_metrics
from src.fairness.single_attribute import run_single_attribute_audits


def run_phase2_pipeline(config_path="config/default_config.json", output_path="data/processed/baseline_results.json"):
    """
    Execute the Phase 2 pipeline and save results to JSON.

    Parameters:
        config_path (str): Path to configuration file.
        output_path (str): Path to destination JSON file for baseline results.

    Returns:
        dict: Complete dictionary containing baseline ML metrics and fairness audit results.
    """
    print("=" * 80)
    print("PHASE 2: Baseline Model Training & Single-Attribute Fairness Audit")
    print("=" * 80)

    # 1. Load configuration and raw dataset
    config = load_config(config_path)
    print(f"\n[1/5] Loading dataset from {config['dataset']['raw_path']}...")
    raw_df = load_raw_data(config_path)
    print(f"      Raw dataset loaded: {len(raw_df):,} rows.")

    # 2. Preprocess data into X, y, A
    print("\n[2/5] Cleaning missing values and preprocessing features...")
    X, y, A = prepare_pipeline_data(raw_df, config=config)
    print(f"      Feature matrix X shape: {X.shape}")
    print(f"      Target vector y shape: {y.shape}")
    print(f"      Protected attributes A shape: {A.shape}")

    # 3. Train-test split
    print("\n[3/5] Performing reproducible train/test split...")
    splits = split_pipeline_data(X, y, A, config=config)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]
    A_train, A_test = splits["A_train"], splits["A_test"]

    print(f"      Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 4. Model Training & Evaluation
    print("\n[4/5] Training baseline Logistic Regression model...")
    seed = config.get("random_seed", 42)
    model = train_baseline_model(X_train, y_train, random_state=seed, max_iter=1000)
    
    y_pred, y_prob = predict_model(model, X_test)
    performance_metrics = evaluate_performance_metrics(y_test, y_pred, y_prob)

    print("\n--- Baseline ML Performance Metrics ---")
    print(f"Accuracy : {performance_metrics['accuracy']:.4f}")
    print(f"Precision: {performance_metrics['precision']:.4f}")
    print(f"Recall   : {performance_metrics['recall']:.4f}")
    print(f"F1-Score : {performance_metrics['f1_score']:.4f}")
    if performance_metrics['roc_auc'] is not None:
        print(f"ROC-AUC  : {performance_metrics['roc_auc']:.4f}")
    cm = performance_metrics['confusion_matrix']
    print(f"Confusion Matrix: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}")

    # 5. Fairness Auditing
    print("\n[5/5] Performing Single-Attribute Fairness Audits ('sex' and 'race')...")
    fairness_audits = run_single_attribute_audits(y_test, y_pred, A_test)

    for attr, audit in fairness_audits.items():
        print(f"\n--- Single-Attribute Audit: {attr.upper()} ---")
        print("  Group Metrics:")
        for grp, m in audit["groups"].items():
            print(f"    - Group '{grp}' (N={m['sample_count']}):")
            print(f"        Selection Rate : {m['selection_rate']:.4f}")
            print(f"        TPR (Recall)   : {m['true_positive_rate']:.4f}")
            print(f"        FPR            : {m['false_positive_rate']:.4f}")
            print(f"        FNR            : {m['false_negative_rate']:.4f}")
        
        print("  Disparities:")
        for disp_name, disp_val in audit["disparities"].items():
            print(f"    - {disp_name}: {disp_val:.4f}")

    # Save results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_data = {
        "phase": 2,
        "model_type": "LogisticRegression",
        "random_seed": seed,
        "data_summary": {
            "total_raw": len(raw_df),
            "cleaned_total": len(X),
            "train_count": len(X_train),
            "test_count": len(X_test),
            "feature_count": X.shape[1]
        },
        "performance_metrics": performance_metrics,
        "fairness_audits": fairness_audits
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults successfully saved to '{output_path}'")
    print("=" * 80)

    return results_data


if __name__ == "__main__":
    run_phase2_pipeline()
