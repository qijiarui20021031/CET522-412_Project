from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "geolife.sqlite"

# GeoLife root folder containing `Data/000`, `Data/001`, ...
# Place "Geolife Trajectories 1.3" inside the project's data/ folder.
DEFAULT_GEOLIFE_ROOT = PROJECT_ROOT / "data" / "Geolife Trajectories 1.3"

# Only keep a small, clear set of modes for a strong baseline.
# Map "taxi" -> "car".
TARGET_MODES = {"walk", "bike", "bus", "car", "taxi"}

