"""
kalman_filter.py — Kalman Filter for Bounding Box Tracking
===========================================================
This module implements an 8-dimensional linear Kalman filter from scratch
(NumPy only).  The filter estimates where a moving cat will be in the next
frame, which is what makes tracking robust to missed detections and jittery
detector outputs.

State vector (8-D):
    x = [cx, cy, w, h, vcx, vcy, vw, vh]^T

    cx, cy   : box center position (pixels)
    w, h     : box width / height (pixels)
    vcx..vh  : velocity of each of those quantities (pixels per frame)

Measurement vector (4-D) — what the detector gives us each frame:
    z = [cx, cy, w, h]^T

The two key equations (see Cat Project.md / SORT paper):

  Predict (time update):
      x_k|k-1 = F @ x_{k-1}
      P_k|k-1 = F @ P_{k-1} @ F^T + Q

  Update (measurement update):
      K = P H^T (H P H^T + R)^-1        <- Kalman gain
      x_k|k = x_k|k-1 + K (z - H x_k|k-1)
      P_k|k = (I - K H) P_k|k-1

Intuition:
  - F encodes "position += velocity * dt" physics.
  - Q (process noise) says "the world can change unpredictably" — cats
    accelerate, stop, turn.
  - R (measurement noise) says "the detector is good but not perfect".
  - The Kalman gain K blends prediction vs measurement optimally: when the
    detector reports a detection we trust it proportionally to how uncertain
    our prediction was.

The filter itself is stateless: every method takes (mean, covariance) and
returns new ones. This keeps the design functional and easy to unit test —
the `Track` class (sort.py) owns a track's mean/covariance.
"""

import numpy as np


# Chi-square 95% quantiles used for gating (Mahalanobis outlier rejection).
# Key = degrees of freedom (= dimension of the measurement space, here 4).
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
}


# ─── Box format conversion helpers ────────────────────────────────
def xyxy_to_cxcyah(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from [x1, y1, x2, y2] to center format [cx, cy, w, h].

    Args:
        boxes (ndarray): [..., 4] in corner format

    Returns:
        ndarray: [..., 4] in center format
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    cx = (boxes[..., 0] + boxes[..., 2]) / 2.0
    cy = (boxes[..., 1] + boxes[..., 3]) / 2.0
    w  = boxes[..., 2] - boxes[..., 0]
    h  = boxes[..., 3] - boxes[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def cxcyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    """
    Convert center-format boxes [cx, cy, w, h] back to [x1, y1, x2, y2].

    Args:
        state (ndarray): [..., 4] center-format (or an 8-D state whose first
                         4 entries are used)

    Returns:
        ndarray: [..., 4] corner format
    """
    state = np.asarray(state, dtype=np.float64)[..., :4]
    cx, cy, w, h = (
        state[..., 0], state[..., 1], state[..., 2], state[..., 3],
    )
    return np.stack([cx - w / 2.0, cy - h / 2.0,
                     cx + w / 2.0, cy + h / 2.0], axis=-1)


# ─── Kalman Filter ─────────────────────────────────────────────────
class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter for bounding boxes.

    Design notes (following the original SORT/DeepSORT conventions):
      - Noise magnitudes are scaled by the current box height h: bigger
        objects move more pixels per frame AND have noisier detections, so
        both process noise (Q) and measurement noise (R) grow with size.
      - dt = 1 frame per step (we run predict() exactly once per video frame).
    """

    def __init__(self):
        self.dt = 1.0

        # Relative noise weights (from the SORT paper's reference code)
        self._std_weight_position = 1.0 / 20.0   # how noisy positions are
        self._std_weight_velocity = 1.0 / 160.0  # velocities are less noisy

        # ── Motion model F: x' = F x ──────────────────────────────
        # Identity plus dt coupling between positions and velocities:
        #   cx' = cx + vcx*dt,  cy' = cy + vcy*dt,  etc.
        self._motion_mat = np.eye(8)
        for i in range(4):
            self._motion_mat[i, i + 4] = self.dt

        # ── Observation model H: z = H x ───────────────────────────
        # We only observe the first 4 state entries (positions/sizes).
        self._update_mat = np.eye(4, 8)

    def initiate(self, measurement: np.ndarray):
        """
        Create a new track state from an unassociated detection.

        Velocities start at 0 (unknown); the covariance expresses our large
        initial uncertainty so early detections pull the estimate strongly.

        Args:
            measurement (ndarray): [cx, cy, w, h] detection in center format

        Returns:
            (mean, covariance): mean shape [8], covariance shape [8, 8]
        """
        measurement = np.asarray(measurement, dtype=np.float64)

        mean_pos = measurement                       # [cx, cy, w, h]
        mean_vel = np.zeros_like(mean_pos)           # unknown velocities
        mean = np.concatenate([mean_pos, mean_vel])

        h = measurement[3]

        # Position uncertainty: 2x base weight; velocity uncertainty: 10x,
        # because a single observation tells us nothing about motion.
        stds = [
            2 * self._std_weight_position * h,   # cx
            2 * self._std_weight_position * h,   # cy
            2 * self._std_weight_position * h,   # w
            2 * self._std_weight_position * h,   # h
            10 * self._std_weight_velocity * h,  # vcx
            10 * self._std_weight_velocity * h,  # vcy
            10 * self._std_weight_velocity * h,  # vw
            10 * self._std_weight_velocity * h,  # vh
        ]
        covariance = np.diag(np.square(stds))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Run the predict step: project state (and uncertainty) one frame ahead.

        Args:
            mean       (ndarray): [8] current state estimate
            covariance (ndarray): [8, 8] current covariance matrix P

        Returns:
            (new_mean, new_covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],  # scale by height h
            self._std_weight_position * mean[3],
            1e-2,                                  # w/h change slowly; small fixed noise
            1e-2,
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            1e-5,
        ]
        # Q = process noise covariance ("how much can the world surprise us")
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = np.dot(self._motion_mat, mean)                    # F @ x
        covariance = (
            np.dot(np.dot(self._motion_mat, covariance), self._motion_mat.T)
            + motion_cov                                          # F P F^T + Q
        )
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Project state distribution from state space (8-D) into measurement
        space (4-D), adding measurement noise R.

        Returns:
            (projected_mean, projected_covariance): shapes [4] and [4, 4]
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            1e-2,
        ]
        innovation_cov = np.diag(np.square(std))                 # R

        mean_proj = np.dot(self._update_mat, mean)               # H @ x
        cov_proj = (
            np.dot(np.dot(self._update_mat, covariance), self._update_mat.T)
            + innovation_cov                                     # H P H^T + R
        )
        return mean_proj, cov_proj

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray):
        """
        Run the update step: correct the predicted state with a new detection.

        This is where the magic happens:
          innovation  = z - H x        (prediction error)
          kalman_gain = P H^T S^-1     (optimal blending weight)
          x_new       = x + K @ innovation
          P_new       = (I - K H) P

        Args:
            mean        (ndarray): [8] predicted state (from predict())
            covariance  (ndarray): [8, 8] predicted covariance
            measurement (ndarray): [cx, cy, w, h] new detector output

        Returns:
            (new_mean, new_covariance)
        """
        measurement = np.asarray(measurement, dtype=np.float64)

        projected_mean, projected_cov = self.project(mean, covariance)

        # Solve instead of explicit inverse: numerically stable version of
        # K = P H^T (H P H^T + R)^-1
        #   K^T = solve(S^T, (P H^T)^T)
        kalman_gain = np.linalg.solve(
            (projected_cov + 1e-9 * np.eye(4)).T,
            np.dot(covariance, self._update_mat.T).T,
        ).T

        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - kalman_gain.dot(projected_cov).dot(kalman_gain.T)
        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray,
                        gated: bool = True) -> np.ndarray:
        """
        Compute squared Mahalanobis distance between the predicted state and
        candidate measurements.

        d^2 = (z - H x)^T S^-1 (z - H x),   S = H P H^T + R

        This measures "how plausible is this detection given my motion
        model", accounting for both position offset AND estimation
        uncertainty (a far-away detection may still be plausible if we are
        very uncertain).

        Used by DeepSORT for appearance-stage gating: pairs with d^2 above
        the chi-square threshold are considered physically impossible and
        are never matched.

        Args:
            mean         (ndarray): [8] track state mean
            covariance   (ndarray): [8, 8] track state covariance
            measurements (ndarray): [N, 4] candidate detections (center fmt)
            gated        (bool):    if True, distances above the 95% chi-square
                                    threshold become +inf (hard gate)

        Returns:
            ndarray: [N] squared Mahalanobis distances
        """
        measurements = np.atleast_2d(np.asarray(measurements, dtype=np.float64))

        projected_mean, projected_cov = self.project(mean, covariance)

        # Vectorized: diff = Z - mu  ->  [N, 4]
        diff = measurements - projected_mean

        # Stable solve with Cholesky factorization of S
        L = np.linalg.cholesky(projected_cov + 1e-9 * np.eye(4))
        # Solve L y = diff^T  =>  y = L^-1 diff^T
        z = np.linalg.solve(L, diff.T)                # [4, N]
        squared_maha = np.sum(z * z, axis=0)          # [N]

        if gated:
            limit = CHI2INV95[4]
            squared_maha[squared_maha > limit] = np.inf

        return squared_maha


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    kf = KalmanBoxFilter()

    # Simulate a cat walking right at constant velocity: starts at
    # center (100, 200), moves (+12, +3) px/frame, box 80x60 px.
    true_centers = np.array(
        [[100 + 12 * t, 200 + 3 * t] for t in range(30)]
    )
    NOISE_STD = 3.0
    noisy_dets = true_centers + rng.normal(0, NOISE_STD, true_centers.shape)
    measurements = np.concatenate(
        [noisy_dets, np.full((30, 2), [80.0, 60.0])], axis=1
    )  # [cx, cy, w=80, h=60]

    # Initialize from the first detection
    mean, cov = kf.initiate(measurements[0])
    initial_cov_trace = cov.trace()

    pred_errors, filt_errors = [], []
    for t in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        # Prediction error BEFORE seeing this frame's detection
        pred_err = np.linalg.norm(cxcyah_to_xyxy(mean)[:2] - true_centers[t])
        mean, cov = kf.update(mean, cov, measurements[t])
        filt_err = np.linalg.norm(cxcyah_to_xyxy(mean)[:2] - true_centers[t])
        pred_errors.append(pred_err)
        filt_errors.append(filt_err)

    print("Kalman filter sanity test (constant-velocity trajectory)")
    print(f"  Mean raw-detection error:      {NOISE_STD * np.sqrt(2):.2f} px")
    print(f"  Mean predicted error (last 15):"
          f" {np.mean(pred_errors[-15:]):.2f} px")
    print(f"  Mean filtered error (last 15): "
          f"{np.mean(filt_errors[-15:]):.2f} px")
    assert np.mean(filt_errors[-15:]) < NOISE_STD, \
        "Filtered estimate should beat raw noisy detections"
    assert cov.trace() < 0.5 * initial_cov_trace, \
        "Covariance should shrink substantially after repeated updates"

    # Gating test: consistent detection close, impostor far away
    near = measurements[-1].copy()
    far = near.copy()
    far[:2] += 500.0
    d = kf.gating_distance(mean, cov, np.stack([near, far]), gated=False)
    print(f"  Mahalanobis: near={d[0]:.2f}, far={d[1]:.2f}")
    assert d[0] < CHI2INV95[4], "Consistent det should pass gate"
    assert d[1] > CHI2INV95[4], "Impostor det should fail gate"

    print("kalman_filter.py sanity checks passed!")
