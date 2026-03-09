from __future__ import annotations

import numpy as np


EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance (meters) between consecutive points.
    Inputs are in degrees.
    """
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def heading_deg(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Bearing from point1 to point2, in degrees [0, 360).
    """
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    brng = np.rad2deg(np.arctan2(y, x))
    return (brng + 360.0) % 360.0


def abs_smallest_angle_diff_deg(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Smallest absolute angular difference between headings (degrees), in [0, 180].
    """
    diff = (a2 - a1 + 180.0) % 360.0 - 180.0
    return np.abs(diff)

