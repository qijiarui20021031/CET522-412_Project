from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LabelInterval:
    interval_id: int
    start_ts: pd.Timestamp  # tz-aware (UTC)
    end_ts: pd.Timestamp  # tz-aware (UTC)
    mode: str


def list_labeled_users(data_root: Path) -> list[str]:
    """
    GeoLife labeled users have `labels.txt` in the user folder.
    `data_root` should point to `<GeoLife>/Data`.
    """
    users: list[str] = []
    for user_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if (user_dir / "labels.txt").exists():
            users.append(user_dir.name)
    return users


def read_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path, sep="\t")
    # Expected columns: Start Time, End Time, Transportation Mode
    df = df.rename(
        columns={
            "Start Time": "start_ts",
            "End Time": "end_ts",
            "Transportation Mode": "mode",
        }
    )
    df["start_ts"] = pd.to_datetime(df["start_ts"], format="%Y/%m/%d %H:%M:%S", utc=True)
    df["end_ts"] = pd.to_datetime(df["end_ts"], format="%Y/%m/%d %H:%M:%S", utc=True)
    df["mode"] = df["mode"].astype(str).str.strip().str.lower()
    df = df.sort_values(["start_ts", "end_ts"]).reset_index(drop=True)
    return df


def read_trajectory_plt(plt_path: Path) -> pd.DataFrame:
    """
    Read one PLT file. Lines 1-6 are headers and ignored.
    Returns DataFrame with columns: ts (UTC), lat, lon, altitude_ft.
    """
    df = pd.read_csv(
        plt_path,
        skiprows=6,
        header=None,
        names=["lat", "lon", "zero", "altitude_ft", "date_days", "date_str", "time_str"],
    )
    df["ts"] = pd.to_datetime(df["date_str"].astype(str) + " " + df["time_str"].astype(str), utc=True)
    df = df[["ts", "lat", "lon", "altitude_ft"]].dropna()
    df = df.sort_values("ts").reset_index(drop=True)
    return df

