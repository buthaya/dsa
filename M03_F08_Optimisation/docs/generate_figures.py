"""Generate the figures embedded in 00_finding_the_best_model.md.

Run from M03_F08_Optimisation/ with: uv run python docs/generate_figures.py

The loss-comparison and threshold figures are built from real numbers computed on
data/01_raw/claims.csv (not the MRH/TP dataset, and not synthetic placeholders), so the
narrative numbers quoted in the markdown doc match what this script prints.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
DATA_PATH = Path(__file__).parent.parent / "data" / "01_raw" / "claims.csv"

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Load & clean claims.csv
# ---------------------------------------------------------------------------
def load_claims() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col=0)
    df["claim_amount"] = (
        df["claim_amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    )
    df["claim_amount"] = pd.to_numeric(df["claim_amount"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Phase 0a/0b: severity by claim_type, global-mean vs per-segment loss
# ---------------------------------------------------------------------------
def analyze_severity(df: pd.DataFrame) -> dict:
    severity = df.dropna(subset=["claim_amount"])
    by_type = severity.groupby("claim_type")["claim_amount"].agg(["mean", "median", "count"])
    print("Claim amount by claim_type:")
    print(by_type)

    y = severity["claim_amount"].to_numpy()
    global_mean = y.mean()
    group_mean = severity.groupby("claim_type")["claim_amount"].transform("mean").to_numpy()

    mse_global = np.mean((y - global_mean) ** 2)
    mse_group = np.mean((y - group_mean) ** 2)
    mae_global = np.mean(np.abs(y - global_mean))
    mae_group = np.mean(np.abs(y - group_mean))

    print(f"\nGlobal-mean-only model: MSE={mse_global:,.0f}  MAE={mae_global:,.0f}")
    print(f"Per-claim_type-mean model: MSE={mse_group:,.0f}  MAE={mae_group:,.0f}")
    print(f"MSE reduction from segmenting: {(1 - mse_group / mse_global):.1%}")
    print(f"MAE reduction from segmenting: {(1 - mae_group / mae_global):.1%}")

    fraud_rate = (df["fraudulent"] == "Yes").mean()
    fraud_by_type = df.groupby("claim_type")["fraudulent"].apply(lambda s: (s == "Yes").mean())
    print(f"\nOverall fraud rate: {fraud_rate:.1%}")
    print("Fraud rate by claim_type:")
    print(fraud_by_type)

    return {
        "by_type": by_type,
        "y": y,
        "global_mean": global_mean,
        "group_mean": group_mean,
        "mse_global": mse_global,
        "mse_group": mse_group,
        "mae_global": mae_global,
        "mae_group": mae_group,
        "fraud_rate": fraud_rate,
        "fraud_by_type": fraud_by_type,
    }


# ---------------------------------------------------------------------------
# Phase 0c: threshold sweep for the fraudulent classifier
# ---------------------------------------------------------------------------
def analyze_threshold(df: pd.DataFrame) -> dict:
    features = df[["claim_area", "claim_type", "police_report", "total_policy_claims"]].copy()
    features["claim_amount"] = df["claim_amount"]
    features = features.dropna()
    target = (df.loc[features.index, "fraudulent"] == "Yes").astype(int)

    cat_cols = ["claim_area", "claim_type", "police_report"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(features[cat_cols])
    X_num = features[["total_policy_claims", "claim_amount"]].to_numpy()
    X = np.hstack([X_num, X_cat])
    y = target.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    precisions, recalls, accuracies, f1s = [], [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        precisions.append(precision_score(y_test, pred, zero_division=0))
        recalls.append(recall_score(y_test, pred, zero_division=0))
        accuracies.append(accuracy_score(y_test, pred))
        f1s.append(f1_score(y_test, pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    print(f"\nThreshold sweep: base rate in test set = {y_test.mean():.1%}")
    print(f"Best F1 at threshold={thresholds[best_idx]:.2f} "
          f"(precision={precisions[best_idx]:.2f}, recall={recalls[best_idx]:.2f}, "
          f"accuracy={accuracies[best_idx]:.2f})")
    print(f"Default threshold=0.50 -> precision={precisions[9]:.2f}, recall={recalls[9]:.2f}, "
          f"accuracy={accuracies[9]:.2f}")

    return {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "accuracies": accuracies,
        "f1s": f1s,
        "best_idx": best_idx,
    }


# ---------------------------------------------------------------------------
# Figure 1: bias-variance tradeoff (synthetic polynomial regression)
# ---------------------------------------------------------------------------
def make_bias_variance_figure() -> None:
    n = 30
    x = np.sort(rng.uniform(-1, 1, n))
    y_true = 1.5 * x**3 - 0.5 * x
    y = y_true + rng.normal(scale=0.15, size=n)

    x_test = np.sort(rng.uniform(-1, 1, 200))
    y_test_true = 1.5 * x_test**3 - 0.5 * x_test
    y_test = y_test_true + rng.normal(scale=0.15, size=200)

    degrees = range(1, 16)
    train_errors, test_errors = [], []
    for d in degrees:
        coeffs = np.polyfit(x, y, d)
        train_pred = np.polyval(coeffs, x)
        test_pred = np.polyval(coeffs, x_test)
        train_errors.append(np.mean((y - train_pred) ** 2))
        test_errors.append(np.mean((y_test - test_pred) ** 2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(list(degrees), train_errors, marker="o", label="Train error")
    ax.plot(list(degrees), test_errors, marker="o", label="Test error")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree (model complexity)")
    ax.set_ylabel("Mean squared error (log scale)")
    ax.set_title("Bias-variance tradeoff: train vs test error")
    ax.legend()
    ax.axvline(degrees[int(np.argmin(test_errors))], color="gray", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "bias_variance_tradeoff.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: loss function shapes + real claims.csv segmentation effect
# ---------------------------------------------------------------------------
def make_loss_comparison_figure(severity_stats: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    residual = np.linspace(-5, 5, 200)
    mse = residual**2
    mae = np.abs(residual)
    delta = 1.5
    huber = np.where(np.abs(residual) <= delta, 0.5 * residual**2, delta * (np.abs(residual) - 0.5 * delta))

    ax = axes[0]
    ax.plot(residual, mse, label="Squared error (L2)")
    ax.plot(residual, mae, label="Absolute error (L1)")
    ax.plot(residual, huber, label="Huber (delta=1.5)")
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss shape vs residual")
    ax.legend()

    ax = axes[1]
    labels = ["Global mean\n(no claim_type)", "Per claim_type\nmean"]
    mse_vals = [severity_stats["mse_global"], severity_stats["mse_group"]]
    mae_vals = [severity_stats["mae_global"], severity_stats["mae_group"]]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, mse_vals, width, label="MSE")
    ax.bar(x_pos + width / 2, mae_vals, width, label="MAE")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (claims.csv, claim_amount)")
    ax.set_title("Effect of segmenting by claim_type")
    ax.legend()

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "loss_functions_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: threshold tradeoff (real classifier sweep on claims.csv)
# ---------------------------------------------------------------------------
def make_threshold_figure(threshold_stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = threshold_stats["thresholds"]
    ax.plot(t, threshold_stats["precisions"], marker=".", label="Precision")
    ax.plot(t, threshold_stats["recalls"], marker=".", label="Recall")
    ax.plot(t, threshold_stats["accuracies"], marker=".", label="Accuracy")
    ax.plot(t, threshold_stats["f1s"], marker=".", label="F1", linestyle="--")
    ax.axvline(t[threshold_stats["best_idx"]], color="gray", linestyle=":", label="Best F1 threshold")
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.4, label="Default 0.50 threshold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Precision/recall/accuracy vs threshold (claims.csv, fraudulent)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "threshold_tradeoff.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_claims()
    severity_stats = analyze_severity(df)
    threshold_stats = analyze_threshold(df)

    make_bias_variance_figure()
    make_loss_comparison_figure(severity_stats)
    make_threshold_figure(threshold_stats)
    print(f"\nFigures written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
