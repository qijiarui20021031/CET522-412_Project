from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_DB_PATH  # noqa: E402


st.set_page_config(page_title="GeoLife Mode Classification", layout="wide")


@st.cache_resource
def load_model(model_path: Path):
    obj = joblib.load(model_path)
    return obj["model"], obj["feature_columns"], obj["labels"]


@st.cache_data
def load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@st.cache_data
def query_df(db_path: Path, sql: str, params=()) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def page_background() -> None:
    st.title("GeoLife Transportation Mode Classification")
    st.write(
        """
This project uses the Microsoft Research **GeoLife Trajectories 1.3** dataset
to build a transportation mode classifier based on GPS trajectories.

We extract trip segments from raw trajectories using the provided
mode labels, engineer features (speed, acceleration, dwell time,
sampling interval, heading change, etc.), and train a baseline model.
An interactive Streamlit app is used to explore the data and predictions.
        """
    )
    st.subheader("Research questions")
    st.write(
        "- Can simple segment‑level statistics (speed, stops, turns, sampling interval, etc.)\n"
        "  distinguish between `walk / bike / bus / car`?\n"
        "- How well does the model generalize **across users** when we split train/test by user ID?"
    )
    st.subheader("Data and license")
    st.write(
        "GeoLife is released for **non‑commercial use only**. "
        "Please do not publicly redistribute the raw data or any derived data."
    )


def page_data_and_db(db_path: Path) -> None:
    st.header("Data and data management (SQLite)")
    st.write(f"Current database path: `{db_path}`")

    st.subheader("Data limitations")
    st.write(
        "- **Labeled users only**: We use only users who have a `labels.txt` file; unlabeled users are excluded.\n"
        "- **Mode mapping**: Taxi is mapped to car; only four modes are retained: walk, bike, bus, car.\n"
        "- **Sampling**: GeoLife uses irregular GPS sampling; we downsample points for visualization.\n"
        "- **Geographic scope**: Data is from Beijing and surrounding areas; results may not generalize to other regions."
    )

    counts = {
        "users": "SELECT COUNT(*) AS n FROM users",
        "label_intervals": "SELECT COUNT(*) AS n FROM label_intervals",
        "segments": "SELECT COUNT(*) AS n FROM segments",
        "segment_points": "SELECT COUNT(*) AS n FROM segment_points",
        "segment_features": "SELECT COUNT(*) AS n FROM segment_features",
    }
    cols = st.columns(len(counts))
    for i, (name, sql) in enumerate(counts.items()):
        n = int(query_df(db_path, sql)["n"].iloc[0])
        cols[i].metric(name, n)

    st.subheader("Schema (simplified relational model)")
    st.code(
        """
users(user_id)
label_intervals(user_id, start_ts, end_ts, mode)
segments(segment_id, user_id, start_ts, end_ts, true_mode, n_points, distance_m, duration_s, avg_speed_mps, created_from)
segment_points(segment_id, seq, ts, lat, lon)  # downsample for visualization
segment_features(segment_id, user_id, true_mode, ...features...)
        """.strip()
    )

    st.subheader("Mode distribution (segments)")
    dist = query_df(
        db_path,
        "SELECT true_mode, COUNT(*) AS n FROM segments GROUP BY true_mode ORDER BY n DESC",
    )
    fig = px.bar(dist, x="true_mode", y="n", title="Segments by mode")
    st.plotly_chart(fig, use_container_width=True)


def page_model_results(db_path: Path, metrics_path: Path) -> None:
    st.header("Model results (baseline)")
    if not metrics_path.exists():
        st.warning("Metrics file not found. Please run `python pipeline/train_model.py` first.")
        return
    m = load_metrics(metrics_path)
    st.write(f"- segments: **{m['n_segments']}**")
    st.write(f"- users: **{m['n_users']}**")

    labels = m["labels"]
    cm = pd.DataFrame(m["confusion_matrix"], index=labels, columns=labels)
    fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix (test)")
    st.plotly_chart(fig, use_container_width=True)

    rep = m["classification_report"]
    summary_rows = []
    for k in labels + ["macro avg", "weighted avg"]:
        if k in rep:
            summary_rows.append(
                {
                    "label": k,
                    "precision": rep[k].get("precision"),
                    "recall": rep[k].get("recall"),
                    "f1": rep[k].get("f1-score"),
                    "support": rep[k].get("support"),
                }
            )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


def page_interactive(db_path: Path, model_path: Path) -> None:
    st.header("Segment playground: pick a trip → guess the mode")
    if not model_path.exists():
        st.warning("Model file not found. Please run `python pipeline/train_model.py` first.")
        return

    model, feature_cols, _labels = load_model(model_path)

    users = query_df(db_path, "SELECT user_id FROM users ORDER BY user_id")["user_id"].tolist()
    if not users:
        st.warning("Database is empty. Please run `python pipeline/build_db.py` first.")
        return

    user_id = st.selectbox("Select user (user_id)", users, index=0)

    segs = query_df(
        db_path,
        """
        SELECT segment_id, start_ts, end_ts, true_mode, distance_m, duration_s, avg_speed_mps
        FROM segments
        WHERE user_id=?
        ORDER BY start_ts
        """,
        (user_id,),
    )
    if segs.empty:
        st.info("No segment under this filter.")
        return

    st.markdown("### Segment selection")
    all_ids_str = segs["segment_id"].astype(str)
    min_id = int(segs["segment_id"].min())
    max_id = int(segs["segment_id"].max())
    total_segments = len(segs)
    st.caption(
        f"User {user_id} has {total_segments} segments. "
        f"Available segment_id range: [{min_id}, {max_id}]. "
        "Type a numeric prefix to narrow down the list (e.g. 1, 12, 123)."
    )
    prefix = st.text_input("Segment_id prefix (start typing to search)", value="", max_chars=10)
    if not prefix:
        st.info("Type a numeric prefix above to see matching segment_ids for this user.")
        return

    candidates = all_ids_str[all_ids_str.str.startswith(prefix)]
    if candidates.empty:
        st.info("No segment_id starts with this prefix. Try another prefix.")
        return

    seg_id_str = st.selectbox("Choose segment_id", sorted(candidates.unique().tolist()))
    seg_id = int(seg_id_str)
    row = segs[segs["segment_id"] == seg_id].iloc[0].to_dict()

    # Always show the trajectory preview for the currently selected segment.
    pts = query_df(
        db_path,
        "SELECT lat, lon, seq FROM segment_points WHERE segment_id=? ORDER BY seq",
        (int(seg_id),),
    )
    if pts.empty:
        st.warning("This segment has no points for visualization.")
        return

    st.markdown("### Trajectory preview")
    center = {"lat": float(pts["lat"].mean()), "lon": float(pts["lon"].mean())}
    layer = pdk.Layer(
        "PathLayer",
        data=[{"path": pts[["lon", "lat"]].values.tolist()}],
        get_path="path",
        get_width=3,
        width_min_pixels=2,
        get_color=[20, 120, 200],
    )
    view_state = pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=11)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    # Only run the prediction pipeline when the user explicitly clicks the button.
    feat = None
    if st.button("Predict for selected segment"):
        feat = query_df(
            db_path,
            f"SELECT {', '.join(feature_cols)} FROM segment_features WHERE segment_id=?",
            (int(seg_id),),
        )
        if feat.empty:
            st.warning("This segment has no features.")
            feat = None
        else:
            pred = model.predict(feat[feature_cols])[0]
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feat[feature_cols])[0]

            col_meta, col_proba = st.columns([1, 1])
            with col_meta:
                st.markdown("### Ground truth vs prediction")
                st.write(f"True mode: **{row['true_mode']}**  |  Predicted: **{pred}**")
                st.markdown("### Segment summary")
                st.metric("Distance (km)", f"{row['distance_m'] / 1000.0:.2f}")
                st.metric("Duration (min)", f"{row['duration_s'] / 60.0:.1f}")
                st.metric("Avg speed (m/s)", f"{row['avg_speed_mps']:.2f}")

            if proba is not None:
                with col_proba:
                    st.markdown("### Predicted probabilities")
                    prob_df = pd.DataFrame(
                        {"mode": model.classes_, "probability": proba}
                    ).sort_values("probability", ascending=False)
                    fig_prob = px.bar(prob_df, x="mode", y="probability", range_y=[0, 1])
                    st.plotly_chart(fig_prob, use_container_width=True)


def page_quick_interactive(db_path: Path, model_path: Path) -> None:
    """
    Evaluation‑focused page: explore aggregate prediction accuracy
    under simple filters (user and true mode).
    """
    st.header("Accuracy explorer: how well does the model perform?")
    if not model_path.exists():
        st.warning("Model file not found. Please run `python pipeline/train_model.py` first.")
        return

    model, feature_cols, _labels = load_model(model_path)

    users = query_df(db_path, "SELECT user_id FROM users ORDER BY user_id")["user_id"].tolist()
    if not users:
        st.warning("Database is empty. Please run `python pipeline/build_db.py` first.")
        return

    user_options = ["All users"] + users
    user_choice = st.selectbox("Filter by user", user_options, index=0, key="quick_user")

    mode_options = ["All modes", "walk", "bike", "bus", "car"]
    mode_choice = st.selectbox("Filter by true_mode", mode_options, index=0, key="quick_mode")

    feat_df = query_df(
        db_path,
        f"""
        SELECT segment_id, user_id, true_mode, {', '.join(feature_cols)}
        FROM segment_features
        """,
    )
    if feat_df.empty:
        st.info("No features found in database.")
        return

    mask = pd.Series(True, index=feat_df.index)
    if user_choice != "All users":
        mask &= feat_df["user_id"] == user_choice
    if mode_choice != "All modes":
        mask &= feat_df["true_mode"] == mode_choice

    subset = feat_df[mask].reset_index(drop=True)
    if subset.empty:
        st.info("No segments match the current filter.")
        return

    X_sub = subset[feature_cols]
    y_true = subset["true_mode"].astype(str)
    y_pred = model.predict(X_sub)

    accuracy = (y_pred == y_true).mean()
    st.markdown("### Overall accuracy")
    st.metric("Accuracy", f"{accuracy:.3f}", help="Fraction of correctly classified segments.")
    st.metric("Number of segments", f"{len(subset)}")

    # Simple per‑class accuracy table (no charts).
    st.markdown("### Per‑class accuracy (within current filter)")
    per_class = (
        pd.DataFrame({"true": y_true, "pred": y_pred})
        .groupby("true")
        .apply(lambda g: (g["pred"] == g["true"]).mean())
        .rename("accuracy")
        .reset_index()
    )
    st.dataframe(per_class, use_container_width=True)


def main() -> None:
    db_path = Path(st.sidebar.text_input("SQLite DB path", str(DEFAULT_DB_PATH))).expanduser()
    model_path = PROJECT_ROOT / "models" / "mode_classifier.joblib"
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"

    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Page",
        [
            "Background & motivation",
            "Data & database",
            "Model results",
            "Segment playground (per‑trip demo)",
            "Accuracy explorer (by filter)",
            "Summary",
        ],
        index=0,
    )

    if st.sidebar.button("Exit app"):
        st.sidebar.write("Shutting down server...")
        os._exit(0)

    if page == "Background & motivation":
        page_background()
    elif page == "Data & database":
        page_data_and_db(db_path)
    elif page == "Model results":
        page_model_results(db_path, metrics_path)
    elif page == "Segment playground (per‑trip demo)":
        page_interactive(db_path, model_path)
    elif page == "Accuracy explorer (by filter)":
        page_quick_interactive(db_path, model_path)
    else:
        st.header("Summary")
        st.write(
            "- We convert GeoLife label intervals into trip segments and engineer segment‑level features.\n"
            "- We train a baseline MLP classifier to recognize `walk / bike / bus / car`.\n"
            "- A Streamlit app provides interactive visualizations and prediction demos."
        )


if __name__ == "__main__":
    main()

