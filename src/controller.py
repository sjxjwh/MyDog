"""Controllers for foot trajectory tracking.

Provides inverse-kinematics-based position controllers that map desired
Cartesian foot positions to joint commands for MuJoCo actuators.
"""

from typing import Optional

import numpy as np

from .kinematics import LegKinematics
from .simulator import MuJoCoSim


class IKFootController:
    """Inverse-kinematics-based foot position controller.

    Given a desired foot position in the hip frame, uses numerical IK to
    compute target joint angles and applies them via position control.

    For a specific leg (e.g., 'FR'), the controller:
      1. Transforms the world-frame target to the hip frame.
      2. Solves IK for joint angles.
      3. Sets actuator position targets.
    """

    def __init__(self, sim: MuJoCoSim, leg_prefix: str,
                 kinematics: LegKinematics,
                 kp: float = 50.0, kd: float = 2.0):
        """Initialize the controller.

        Args:
            sim: MuJoCo simulation wrapper.
            leg_prefix: Leg identifier ('FR', 'FL', 'RR', 'RL').
            kinematics: LegKinematics instance for this leg.
            kp: Position gain for PD control (applied in MuJoCo via actuator gain).
            kd: Velocity damping gain.
        """
        self._sim = sim
        self._leg = leg_prefix
        self._kin = kinematics
        self._kp = kp
        self._kd = kd

        # Identify joint/actuator/site names for this leg
        self._find_names()
        self._joint_names = self._joint_qpos_names

    def _find_names(self):
        """Find joint, actuator, and site names for this leg.

        Populates:
          - self._ctrl_names: actuator names for setting control targets
          - self._joint_qpos_names: joint names for reading qpos values
          - self._foot_site_name: name of the foot site for position feedback
        """
        base_names = ["hip", "thigh", "calf"]

        self._ctrl_names = []
        self._joint_qpos_names = []

        for base in base_names:
            act_name = f"{self._leg}_{base}"          # e.g., FR_hip
            joint_name = f"{self._leg}_{base}_joint"  # e.g., FR_hip_joint

            # Actuator name: try both conventions
            if act_name in self._sim._actuator_ids:
                self._ctrl_names.append(act_name)
            elif joint_name in self._sim._actuator_ids:
                self._ctrl_names.append(joint_name)
            else:
                self._ctrl_names.append(act_name)  # fallback

            # Joint qpos name: prefer _joint suffix
            if joint_name in self._sim._joint_ids:
                self._joint_qpos_names.append(joint_name)
            elif act_name in self._sim._joint_ids:
                self._joint_qpos_names.append(act_name)
            else:
                self._joint_qpos_names.append(joint_name)  # fallback

        # Find foot site
        foot_candidates = [
            self._leg,                    # FR (MJCF convention)
            f"{self._leg}_foot",          # FR_foot
            f"{self._leg}_foot_site",
        ]
        self._foot_site_name = None
        for c in foot_candidates:
            if c in self._sim._site_ids:
                self._foot_site_name = c
                break
        # Fallback: search all sites for ones starting with leg prefix
        if self._foot_site_name is None:
            for site_name in self._sim._site_ids:
                if site_name.startswith(self._leg) and "foot" in site_name.lower():
                    self._foot_site_name = site_name
                    break

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    def get_hip_frame_position(self) -> np.ndarray:
        """Get the world-frame position of the hip body."""
        hip_body = f"{self._leg}_hip"
        try:
            return self._sim.get_body_position(hip_body)
        except KeyError:
            # Fallback: try to compute from trunk
            trunk_pos = self._sim.get_body_position("trunk")
            # Approximate hip offset from MJCF
            x_sign = 1 if self._leg[0] == "F" else -1  # Front vs Rear
            y_sign = 1 if self._leg[1] == "L" else -1  # Left vs Right
            hip_offset = np.array([0.1881 * x_sign, 0.04675 * y_sign, 0.0])
            return trunk_pos + hip_offset

    def world_to_hip_frame(self, world_pos: np.ndarray) -> np.ndarray:
        """Transform a world-frame position to the hip frame."""
        hip_pos = self.get_hip_frame_position()
        hip_rot = self._sim.get_body_rotation(f"{self._leg}_hip")
        return hip_rot.T @ (world_pos - hip_pos)

    def hip_to_world_frame(self, hip_pos: np.ndarray) -> np.ndarray:
        """Transform a hip-frame position to the world frame."""
        origin = self.get_hip_frame_position()
        hip_rot = self._sim.get_body_rotation(f"{self._leg}_hip")
        return origin + hip_rot @ hip_pos

    def solve_ik(self, target_world: np.ndarray,
                 q0: Optional[np.ndarray] = None,
                 max_iter: int = 100, tol: float = 1e-5,
                 damping: float = 1e-3) -> Optional[np.ndarray]:
        """Solve IK for a world-frame target foot position using MuJoCo FK.

        Uses Gauss-Newton with damped least squares. Computes Jacobian via
        finite differences on the actual MuJoCo model, ensuring consistent
        kinematics with the simulator.

        Returns joint angles or None if unreachable.
        """
        target = np.asarray(target_world, dtype=float)

        if q0 is None:
            q = self.get_current_joint_angles()
        else:
            q = np.asarray(q0, dtype=float).copy()

        eps = 1e-6

        for iteration in range(max_iter):
            # Set joint angles and compute FK via MuJoCo
            for jname, angle in zip(self._joint_qpos_names, q):
                self._sim.set_qpos(jname, angle)
            self._sim.forward()
            foot = self.get_foot_position()
            error = target - foot

            if np.linalg.norm(error) < tol:
                return q

            # Compute Jacobian via finite differences on MuJoCo FK
            J = np.zeros((3, 3))
            for i in range(3):
                q_pert = q.copy()
                q_pert[i] += eps
                for jname, angle in zip(self._joint_qpos_names, q_pert):
                    self._sim.set_qpos(jname, angle)
                self._sim.forward()
                foot_pert = self.get_foot_position()
                J[:, i] = (foot_pert - foot) / eps

            # Damped least squares: dq = J^T (JJ^T + λ²I)⁻¹ error
            JJT = J @ J.T
            I3 = np.eye(3)
            try:
                dq = J.T @ np.linalg.solve(JJT + damping**2 * I3, error)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(J).dot(error)

            # Line search (simple)
            alpha = 1.0
            for _ in range(10):
                q_new = q + alpha * dq
                for jname, angle in zip(self._joint_qpos_names, q_new):
                    self._sim.set_qpos(jname, angle)
                self._sim.forward()
                foot_new = self.get_foot_position()
                if np.linalg.norm(target - foot_new) < np.linalg.norm(error):
                    break
                alpha *= 0.5

            q = q + alpha * dq

        # Final check
        for jname, angle in zip(self._joint_qpos_names, q):
            self._sim.set_qpos(jname, angle)
        self._sim.forward()
        foot_final = self.get_foot_position()
        if np.linalg.norm(target - foot_final) < tol * 100:
            return q
        return None

    def set_joint_targets(self, q_target: np.ndarray):
        """Set actuator position targets for the leg joints."""
        for i, cname in enumerate(self._ctrl_names):
            if i < len(q_target):
                self._sim.set_joint_ctrl(cname, q_target[i])

    def get_current_joint_angles(self) -> np.ndarray:
        """Get current joint angles for this leg."""
        angles = []
        for jname in self._joint_qpos_names:
            try:
                angles.append(self._sim.get_joint_qpos(jname))
            except KeyError:
                angles.append(0.0)
        return np.array(angles)

    def get_foot_position(self) -> np.ndarray:
        """Get current foot position in world frame.

        Uses foot site if available, otherwise falls back to foot body position.
        """
        if self._foot_site_name:
            return self._sim.get_site_position(self._foot_site_name)
        # Fallback: use foot body
        foot_body = f"{self._leg}_foot"
        try:
            return self._sim.get_body_position(foot_body)
        except KeyError:
            # Last resort: approximate from hip frame + FK
            hip_pos = self.get_hip_frame_position()
            q = self.get_current_joint_angles()
            # Use the analytical FK as approximation
            foot_local, _ = self._kin.forward_kinematics(q)
            hip_rot = np.eye(3)  # approximate
            return hip_pos + hip_rot @ foot_local

    def control(self, target_world: np.ndarray) -> Optional[np.ndarray]:
        """Compute control: solve IK and set joint targets.

        Returns the computed joint angles, or None if IK failed.
        """
        q0 = self.get_current_joint_angles()
        q_target = self.solve_ik(target_world, q0=q0)
        if q_target is not None:
            self.set_joint_targets(q_target)
        return q_target


def create_go1_leg_kinematics(leg_prefix: str) -> LegKinematics:
    """Create a LegKinematics instance matching the Go1 leg dimensions.

    Dimensions extracted from the MuJoCo MJCF go1.xml model:
      - Hip offset from trunk center to hip body
      - Thigh length: 0.213 m (distance from hip pitch to knee pitch)
      - Calf length: 0.213 m (distance from knee pitch to foot)

    Args:
        leg_prefix: Which leg ('FR', 'FL', 'RR', 'RL').

    Returns:
        Configured LegKinematics instance.
    """
    # Hip positions relative to trunk (from MJCF go1.xml)
    x_sign = 1 if leg_prefix[0] == "F" else -1  # Front (0.1881) vs Rear (-0.1881)
    y_sign = 1 if leg_prefix[1] == "L" else -1  # Left (0.04675) vs Right (-0.04675)

    hip_offset = np.array([0.1881 * x_sign, 0.04675 * y_sign, 0.0])

    # Thigh and calf lengths (from MJCF: calf pivot is at z=-0.213 from thigh origin)
    thigh_length = 0.213  # m
    calf_length = 0.213   # m

    # Joint axes (Go1 convention):
    # Joint 0: abduction → X-axis in hip frame
    # Joint 1: hip pitch → Y-axis
    # Joint 2: knee pitch → Y-axis
    joint_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    return LegKinematics(
        hip_offset=hip_offset,
        thigh_length=thigh_length,
        calf_length=calf_length,
        joint_axes=joint_axes,
    )
