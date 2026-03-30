from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GazeMeasurement:
    raw_nx: float
    raw_ny: float
    face_nx: float
    face_ny: float
    yaw: float
    pitch: float
    timestamp: float


@dataclass(slots=True)
class CalibrationSample:
    measurement: GazeMeasurement
    target_nx: float
    target_ny: float
    weight: float = 1.0


@dataclass(slots=True)
class CalibrationProfile:
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_coeffs: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    y_coeffs: np.ndarray

    def map_to_normalized(self, measurement: GazeMeasurement) -> tuple[float, float]:
        x_features = _x_features(measurement)
        y_features = _y_features(measurement)

        x_norm = (x_features - self.x_mean) / self.x_scale
        y_norm = (y_features - self.y_mean) / self.y_scale

        nx = float(np.dot(np.append(x_norm, 1.0), self.x_coeffs))
        ny = float(np.dot(np.append(y_norm, 1.0), self.y_coeffs))

        return (
            float(np.clip(nx, 0.0, 1.0)),
            float(np.clip(ny, 0.0, 1.0)),
        )


def build_profile(
    samples: list[CalibrationSample],
    ridge: float = 1e-3,
) -> CalibrationProfile:
    if len(samples) < 40:
        raise ValueError("Not enough calibration samples collected.")

    x_matrix = np.array([
        _x_features(sample.measurement)
        for sample in samples
    ], dtype=np.float64)
    y_matrix = np.array([
        _y_features(sample.measurement)
        for sample in samples
    ], dtype=np.float64)

    target_x = np.array([sample.target_nx for sample in samples], dtype=np.float64)
    target_y = np.array([sample.target_ny for sample in samples], dtype=np.float64)
    weights = np.array([sample.weight for sample in samples], dtype=np.float64)

    x_mean, x_scale = _normalization_stats(x_matrix)
    y_mean, y_scale = _normalization_stats(y_matrix)

    x_norm = (x_matrix - x_mean) / x_scale
    y_norm = (y_matrix - y_mean) / y_scale

    x_coeffs = _fit_weighted_ridge(x_norm, target_x, weights, ridge)
    y_coeffs = _fit_weighted_ridge(y_norm, target_y, weights, ridge)

    return CalibrationProfile(
        x_mean=x_mean,
        x_scale=x_scale,
        x_coeffs=x_coeffs,
        y_mean=y_mean,
        y_scale=y_scale,
        y_coeffs=y_coeffs,
    )


def _x_features(measurement: GazeMeasurement) -> np.ndarray:
    return np.array([
        measurement.raw_nx,
        measurement.face_nx,
        measurement.yaw,
    ], dtype=np.float64)


def _y_features(measurement: GazeMeasurement) -> np.ndarray:
    return np.array([
        measurement.raw_ny,
        measurement.face_ny,
        measurement.pitch,
    ], dtype=np.float64)


def _normalization_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _fit_weighted_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    design = np.column_stack((features, np.ones(len(features), dtype=np.float64)))
    weight_root = np.sqrt(np.clip(weights, 1e-6, None))
    weighted_design = design * weight_root[:, None]
    weighted_targets = targets * weight_root

    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0

    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_targets
    return np.linalg.solve(lhs, rhs)
