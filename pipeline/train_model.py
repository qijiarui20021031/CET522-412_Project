from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import wandb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_DB_PATH  # noqa: E402


FEATURE_COLUMNS = [
    "duration_s",
    "distance_m",
    "avg_speed_mps",
    "std_speed_mps",
    "p95_speed_mps",
    "max_speed_mps",
    "mean_accel_mps2",
    "std_accel_mps2",
    "stop_ratio",
    "mean_sampling_dt_s",
    "std_sampling_dt_s",
    "mean_abs_heading_change_deg",
]


def load_features(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        f"""
        SELECT
          segment_id, user_id, true_mode,
          {", ".join(FEATURE_COLUMNS)}
        FROM segment_features
        """,
        conn,
    )
    conn.close()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out-model", type=Path, default=PROJECT_ROOT / "models" / "mode_classifier.joblib")
    parser.add_argument("--out-metrics", type=Path, default=PROJECT_ROOT / "models" / "metrics.json")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    if use_wandb:
        wandb.init(
            project="geolife-mode-classification",
            config={
                "model": "MLPClassifier",
                "hidden_layer_sizes": (64, 32),
                "test_size": args.test_size,
                "seed": args.seed,
            },
        )

    df = load_features(args.db)
    if df.empty:
        raise SystemExit("No features found. Run `python pipeline/build_db.py` first.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["true_mode"].astype(str).copy()
    groups = df["user_id"].astype(str).copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    numeric = FEATURE_COLUMNS
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric)],
        remainder="drop",
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        max_iter=500,
        random_state=args.seed,
        early_stopping=False,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    clf_fitted = pipe.named_steps["clf"]
    loss_curve = getattr(clf_fitted, "loss_curve_", None)

    y_pred = pipe.predict(X_test)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": pipe,
            "feature_columns": FEATURE_COLUMNS,
            "labels": labels,
        },
        args.out_model,
    )

    metrics = {
        "db": str(args.db),
        "n_segments": int(df.shape[0]),
        "n_users": int(df["user_id"].nunique()),
        "split": {"test_size": float(args.test_size), "seed": int(args.seed)},
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if use_wandb:
        if loss_curve is not None:
            for step, loss_val in enumerate(loss_curve):
                wandb.log({"train/loss": loss_val}, step=step)
        wandb.log(
            {
                "n_segments": int(df.shape[0]),
                "n_users": int(df["user_id"].nunique()),
                "test/macro_f1": report.get("macro avg", {}).get("f1-score", 0),
                "test/weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
                **{f"test/{k}_f1": report.get(k, {}).get("f1-score", 0) for k in labels if k in report},
            }
        )
        label_to_idx = {c: i for i, c in enumerate(labels)}
        y_true_idx = [label_to_idx.get(y, -1) for y in y_test.tolist()]
        preds_idx = [label_to_idx.get(y, -1) for y in y_pred.tolist()]
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true_idx,
                    preds=preds_idx,
                    class_names=labels,
                )
            }
        )
        wandb.finish()

    print(f"Model saved: {args.out_model}")
    print(f"Metrics saved: {args.out_metrics}")
    print("Test set macro avg f1:", report.get("macro avg", {}).get("f1-score"))


if __name__ == "__main__":
    main()

