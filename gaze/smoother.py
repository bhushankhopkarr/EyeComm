import numpy as np
from filterpy.kalman import KalmanFilter
from config.settings import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE, SMOOTHING_FACTOR
from utils.logger import get_logger

logger = get_logger(__name__)


class GazeSmoother:
    def __init__(self):
        self.kf = self._build_kalman()
        self.initialized = False
        logger.info("GazeSmoother initialized with Kalman filter.")

    def _build_kalman(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)

        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)

        kf.R *= KALMAN_MEASUREMENT_NOISE
        kf.Q *= KALMAN_PROCESS_NOISE
        kf.P *= 10

        return kf

    def smooth(self, x: float, y: float) -> tuple[float, float]:
        if not self.initialized:
            self.kf.x = np.array([[x], [y], [0], [0]], dtype=float)
            self.initialized = True
            return x, y

        self.kf.predict()
        self.kf.update(np.array([[x], [y]], dtype=float))

        return float(self.kf.x[0][0]), float(self.kf.x[1][0])

    def reset(self):
        self.kf = self._build_kalman()
        self.initialized = False
        logger.info("GazeSmoother reset.")


class EWMASmoother:
    def __init__(self):
        self.x = None
        self.y = None

    def smooth(self, x: float, y: float) -> tuple[float, float]:
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = SMOOTHING_FACTOR * self.x + (1 - SMOOTHING_FACTOR) * x
            self.y = SMOOTHING_FACTOR * self.y + (1 - SMOOTHING_FACTOR) * y
        return self.x, self.y

    def reset(self):
        self.x = None
        self.y = None