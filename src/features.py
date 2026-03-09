from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import abs_smallest_angle_diff_deg, haversine_m, heading_deg


@dataclass(frozen=True)
class SegmentFeatures:
    duration_s: float
    distance_m: float
    avg_speed_mps: float
    std_speed_mps: float
    p95_speed_mps: float
    max_speed_mps: float
    mean_accel_mps2: float
    std_accel_mps2: float
    stop_ratio: float
    mean_sampling_dt_s: float
    std_sampling_dt_s: float
    mean_abs_heading_change_deg: float


def compute_segment_features(points: pd.DataFrame) -> SegmentFeatures:
    """
    points: DataFrame with columns [ts, lat, lon] sorted by ts, tz-aware.
    """
    if len(points) < 3:
        raise ValueError("segment needs at least 3 points")

    ts = points["ts"].to_numpy()
    lat = points["lat"].to_numpy(dtype=float)
    lon = points["lon"].to_numpy(dtype=float)

    dt_s = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
    dt_s = dt_s.astype(float)

    # guard against non-positive timestamps
    valid = dt_s > 0
    if valid.sum() < 2:
        raise ValueError("not enough valid dt samples")

    d_m = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    d_m = d_m.astype(float)

    d_m = d_m[valid]
    dt_s = dt_s[valid]

    speed = d_m / dt_s
    # Robustness: clip extreme spikes (keeps airplane-like segments possible but reduces sensor glitches)
    speed = np.clip(speed, 0.0, 120.0)

    duration_s = float((points["ts"].iloc[-1] - points["ts"].iloc[0]).total_seconds())
    distance_m = float(d_m.sum())
    avg_speed_mps = float(distance_m / duration_s) if duration_s > 0 else float(speed.mean())

    std_speed_mps = float(np.std(speed)) if len(speed) > 1 else 0.0
    p95_speed_mps = float(np.percentile(speed, 95))
    max_speed_mps = float(np.max(speed))

    accel = np.diff(speed) / dt_s[1:]
    accel = np.clip(accel, -20.0, 20.0)
    mean_accel_mps2 = float(np.mean(accel)) if len(accel) else 0.0
    std_accel_mps2 = float(np.std(accel)) if len(accel) > 1 else 0.0

    stop_ratio = float(np.mean(speed < 0.5))

    mean_sampling_dt_s = float(np.mean(dt_s))
    std_sampling_dt_s = float(np.std(dt_s)) if len(dt_s) > 1 else 0.0

    hd = heading_deg(lat[:-1][valid], lon[:-1][valid], lat[1:][valid], lon[1:][valid])
    if len(hd) >= 2:
        mean_abs_heading_change_deg = float(np.mean(abs_smallest_angle_diff_deg(hd[:-1], hd[1:])))
    else:
        mean_abs_heading_change_deg = 0.0

    return SegmentFeatures(
        duration_s=duration_s,
        distance_m=distance_m,
        avg_speed_mps=avg_speed_mps,
        std_speed_mps=std_speed_mps,
        p95_speed_mps=p95_speed_mps,
        max_speed_mps=max_speed_mps,
        mean_accel_mps2=mean_accel_mps2,
        std_accel_mps2=std_accel_mps2,
        stop_ratio=stop_ratio,
        mean_sampling_dt_s=mean_sampling_dt_s,
        std_sampling_dt_s=std_sampling_dt_s,
        mean_abs_heading_change_deg=mean_abs_heading_change_deg,
    )


def downsample_points(points: pd.DataFrame, max_points: int = 500) -> pd.DataFrame:
    if len(points) <= max_points:
        out = points.copy()
        out["seq"] = np.arange(len(out), dtype=int)
        return out
    stride = int(np.ceil(len(points) / max_points))
    out = points.iloc[::stride].copy().reset_index(drop=True)
    out["seq"] = np.arange(len(out), dtype=int)
    return out

