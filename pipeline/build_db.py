from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_DB_PATH, DEFAULT_GEOLIFE_ROOT, TARGET_MODES  # noqa: E402
from src.features import compute_segment_features, downsample_points  # noqa: E402
from src.geolife_io import list_labeled_users, read_labels, read_trajectory_plt  # noqa: E402
from src.sqlite_db import connect, init_schema  # noqa: E402


def normalize_mode(mode: str) -> str | None:
    m = str(mode).strip().lower()
    if m not in TARGET_MODES:
        return None
    if m == "taxi":
        return "car"
    return m


def build_for_user(conn, geolife_data_root: Path, user_id: str, *, max_users_debug: int | None = None) -> int:
    user_dir = geolife_data_root / user_id
    labels_path = user_dir / "labels.txt"
    if not labels_path.exists():
        return 0

    labels = read_labels(labels_path)
    labels["mode"] = labels["mode"].map(normalize_mode)
    labels = labels.dropna(subset=["mode"]).reset_index(drop=True)
    if labels.empty:
        return 0

    # Use tz-naive UTC datetime64 for fast slicing/searchsorted.
    labels_start64 = labels["start_ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    labels_end64 = labels["end_ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    conn.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO label_intervals(user_id, start_ts, end_ts, mode) VALUES (?, ?, ?, ?)",
        [
            (user_id, r.start_ts.isoformat(), r.end_ts.isoformat(), r.mode)
            for r in labels.itertuples(index=False)
        ],
    )
    conn.commit()

    # interval_id -> list[DataFrame]
    interval_chunks: dict[int, list[pd.DataFrame]] = {i: [] for i in range(len(labels))}

    traj_dir = user_dir / "Trajectory"
    plt_files = sorted(traj_dir.glob("*.plt"))
    for plt_path in plt_files:
        pts = read_trajectory_plt(plt_path)
        if pts.empty:
            continue
        ts64 = pts["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        file_start64 = ts64[0]
        file_end64 = ts64[-1]

        mask = (labels_end64 >= file_start64) & (labels_start64 <= file_end64)
        if not mask.any():
            continue
        overlaps_idx = np.where(mask)[0]

        for idx in overlaps_idx:
            left = int(np.searchsorted(ts64, labels_start64[idx], side="left"))
            right = int(np.searchsorted(ts64, labels_end64[idx], side="right"))
            if right - left < 3:
                continue
            interval_chunks[int(idx)].append(pts.iloc[left:right][["ts", "lat", "lon"]].copy())

    created_from = "labels.txt + Trajectory/*.plt"
    inserted = 0

    for idx, chunks in interval_chunks.items():
        if not chunks:
            continue
        seg_points = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["ts", "lat", "lon"])
        seg_points = seg_points.sort_values("ts").reset_index(drop=True)
        if len(seg_points) < 10:
            continue

        # Basic QC: remove obviously bad coordinates
        seg_points = seg_points[(seg_points["lat"].between(-90, 90)) & (seg_points["lon"].between(-180, 180))]
        if len(seg_points) < 10:
            continue

        try:
            feats = compute_segment_features(seg_points)
        except Exception:
            continue

        # Filters keep segments meaningful for classification
        if feats.duration_s < 60:
            continue
        if seg_points.shape[0] < 30:
            continue

        true_mode = str(labels.loc[idx, "mode"])

        cur = conn.execute(
            """
            INSERT INTO segments(
              user_id, start_ts, end_ts, true_mode, n_points, distance_m, duration_s, avg_speed_mps, created_from
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                seg_points["ts"].iloc[0].isoformat(),
                seg_points["ts"].iloc[-1].isoformat(),
                true_mode,
                int(seg_points.shape[0]),
                float(feats.distance_m),
                float(feats.duration_s),
                float(feats.avg_speed_mps),
                created_from,
            ),
        )
        segment_id = int(cur.lastrowid)

        ds = downsample_points(seg_points, max_points=500)
        conn.executemany(
            "INSERT INTO segment_points(segment_id, seq, ts, lat, lon) VALUES (?, ?, ?, ?, ?)",
            [
                (segment_id, int(r.seq), r.ts.isoformat(), float(r.lat), float(r.lon))
                for r in ds.itertuples(index=False)
            ],
        )

        conn.execute(
            """
            INSERT INTO segment_features(
              segment_id, user_id, true_mode,
              duration_s, distance_m, avg_speed_mps,
              std_speed_mps, p95_speed_mps, max_speed_mps,
              mean_accel_mps2, std_accel_mps2,
              stop_ratio, mean_sampling_dt_s, std_sampling_dt_s,
              mean_abs_heading_change_deg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                user_id,
                true_mode,
                feats.duration_s,
                feats.distance_m,
                feats.avg_speed_mps,
                feats.std_speed_mps,
                feats.p95_speed_mps,
                feats.max_speed_mps,
                feats.mean_accel_mps2,
                feats.std_accel_mps2,
                feats.stop_ratio,
                feats.mean_sampling_dt_s,
                feats.std_sampling_dt_s,
                feats.mean_abs_heading_change_deg,
            ),
        )
        conn.commit()
        inserted += 1

    return inserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geolife-root", type=Path, default=DEFAULT_GEOLIFE_ROOT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--limit-users", type=int, default=0, help="debug: process only first N users (0=all)")
    args = parser.parse_args()

    geolife_root: Path = args.geolife_root
    geolife_data_root = geolife_root / "Data"
    db_path: Path = args.db

    conn = connect(db_path)
    init_schema(conn)

    users = list_labeled_users(geolife_data_root)
    if args.limit_users and args.limit_users > 0:
        users = users[: args.limit_users]

    total_segments = 0
    for u in users:
        total_segments += build_for_user(conn, geolife_data_root, u)

    print(f"DB: {db_path}")
    print(f"Labeled users processed: {len(users)}")
    print(f"Segments inserted: {total_segments}")


if __name__ == "__main__":
    main()

