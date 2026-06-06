"""Inverse kinematics and Jacobian computation for the robotic dog leg.

Supports arbitrary 3-DOF leg kinematics by computing forward kinematics,
Jacobian matrices, and solving inverse kinematics numerically using the
Gauss-Newton method with Jacobian pseudoinverse.
"""

from typing import Optional

import numpy as np


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Compute the 4x4 homogeneous transformation matrix from DH parameters.

    Args:
        theta: Joint angle (radians).
        d: Link offset.
        a: Link length.
        alpha: Link twist (radians).

    Returns:
        4x4 homogeneous transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])


class LegKinematics:
    """Forward and inverse kinematics for a single 3-DOF leg.

    The leg is modeled as a chain of 3 revolute joints:
      - Joint 0: Hip abduction/adduction (axis ~ X)
      - Joint 1: Hip pitch (axis ~ Y)
      - Joint 2: Knee pitch (axis ~ Y)

    Kinematics are computed using the product-of-exponentials approach with
    numerical Jacobian computation for IK.
    """

    def __init__(self,
                 hip_offset: np.ndarray,
                 thigh_length: float,
                 calf_length: float,
                 joint_axes: Optional[list] = None):
        """Initialize leg kinematics.

        Args:
            hip_offset: Position of the hip joint relative to the body frame [x, y, z].
            thigh_length: Length of the thigh segment.
            calf_length: Length of the calf segment.
            joint_axes: List of 3 rotation axes for hip_abduction, hip_pitch, knee.
                        Default axes match the Go1 convention.
        """
        self.hip_offset = np.asarray(hip_offset, dtype=float)
        self.L1 = thigh_length      # upper leg length
        self.L2 = calf_length       # lower leg length

        # Default: Go1 axes
        # Joint 0: abduction → axis=(1, 0, 0) in body/hip frame
        # Joint 1: hip pitch → axis=(0, 1, 0)
        # Joint 2: knee pitch → axis=(0, 1, 0)
        if joint_axes is None:
            self.axes = [
                np.array([1.0, 0.0, 0.0]),  # abduction
                np.array([0.0, 1.0, 0.0]),  # hip pitch
                np.array([0.0, 1.0, 0.0]),  # knee pitch
            ]
        else:
            self.axes = [np.asarray(ax, dtype=float) for ax in joint_axes]

        # Joint positions relative to hip (in hip frame)
        # The thigh joint is offset from the hip abduction joint
        self.hip_to_thigh = np.array([0.0, 0.0, 0.0])  # no offset for Go1-style
        self.thigh_to_knee = np.array([0.0, 0.0, -self.L1])
        self.knee_to_foot = np.array([0.0, 0.0, -self.L2])

        # In Go1 MJCF:
        # - Hip abduction axis is at the hip body origin
        # - Thigh pivot is at (0, -0.08, 0) from hip (for Front Right), this is after
        #   the abduction rotation, so thigh pitch rotates the thigh+calf
        # - Calf pivot is at (0, 0, -0.213) from thigh

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute foot position and knee position given joint angles.

        Args:
            q: Joint angles [hip_abduction, hip_pitch, knee_pitch] in radians.

        Returns:
            (foot_pos, knee_pos): 3D positions of foot and knee in the hip frame.
        """
        q = np.asarray(q, dtype=float)

        # Joint 0: abduction rotation about axis[0]
        R0 = self._rotation_matrix(self.axes[0], q[0])

        # Thigh pivot: after abduction, the thigh is offset in Y
        thigh_origin = self.hip_to_thigh.copy()

        # Joint 1: hip pitch about axis[1] (after abduction rotation)
        R1 = self._rotation_matrix(self.axes[1], q[1])
        # The axis is rotated by R0 if we're in the world frame, but since
        # we compute in the hip frame with consecutive transforms, we treat
        # each joint in its own local frame.

        # Build forward kinematics using successive transformations
        # Position of knee in hip frame:
        #   knee = R0 * (thigh_offset + R1 * (0, 0, -L1))
        knee_local = np.array([0.0, 0.0, -self.L1])
        knee = R0 @ (thigh_origin + R1 @ knee_local)

        # Joint 2: knee pitch about axis[2]
        R2 = self._rotation_matrix(self.axes[2], q[2])

        # Position of foot in hip frame:
        #   foot = knee + R0 @ R1 @ R2 @ (0, 0, -L2)
        calf_local = np.array([0.0, 0.0, -self.L2])
        foot = knee + R0 @ R1 @ R2 @ calf_local

        return foot, knee

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the 3x3 Jacobian matrix mapping joint velocities to foot
        linear velocity in the hip frame.

        Uses finite differences for generality.
        """
        q = np.asarray(q, dtype=float)
        eps = 1e-6
        J = np.zeros((3, 3))
        f0, _ = self.forward_kinematics(q)
        for i in range(3):
            dq = np.zeros(3)
            dq[i] = eps
            f1, _ = self.forward_kinematics(q + dq)
            J[:, i] = (f1 - f0) / eps
        return J

    def inverse_kinematics(self, target_pos: np.ndarray,
                           q0: Optional[np.ndarray] = None,
                           max_iter: int = 100,
                           tol: float = 1e-6,
                           damping: float = 0.001) -> Optional[np.ndarray]:
        """Solve IK for a target foot position using damped least squares (Levenberg-Marquardt).

        Args:
            target_pos: Desired foot position [x, y, z] in the hip frame.
            q0: Initial guess for joint angles. Defaults to zeros.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance on position error.
            damping: Damping factor for numerical stability.

        Returns:
            Joint angles [q0, q1, q2] if converged, None if failed to converge.
        """
        target = np.asarray(target_pos, dtype=float)
        q = np.zeros(3) if q0 is None else np.asarray(q0, dtype=float).copy()

        for _ in range(max_iter):
            foot, _ = self.forward_kinematics(q)
            error = target - foot

            if np.linalg.norm(error) < tol:
                return q

            J = self.jacobian(q)

            # Damped least squares: dq = J^T (JJ^T + λ²I)⁻¹ error
            JJT = J @ J.T
            I = np.eye(3)
            try:
                dq = J.T @ np.linalg.solve(JJT + damping**2 * I, error)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                dq = np.linalg.pinv(J) @ error

            q += dq

        # Check final error
        foot, _ = self.forward_kinematics(q)
        if np.linalg.norm(target - foot) < tol * 10:
            return q
        return None

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """Compute 3x3 rotation matrix using Rodrigues' formula."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + s * K + (1 - c) * K @ K

    def is_reachable(self, target_pos: np.ndarray, tol: float = 1e-3) -> bool:
        """Check if a target position is within the leg's workspace."""
        dist = np.linalg.norm(target_pos - self.hip_offset)
        max_reach = self.L1 + self.L2
        min_reach = abs(self.L1 - self.L2)
        return min_reach - tol <= dist <= max_reach + tol
