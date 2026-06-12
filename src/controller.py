"""Controllers for foot trajectory tracking.

Provides inverse-kinematics-based position controllers that map desired
Cartesian foot positions to joint commands for MuJoCo actuators.
"""

import ctypes
import ctypes.util
import signal
from typing import Optional

import numpy as np


# ── SIGFPE guard for MuJoCo mj_forward calls ────────────────────────────────
# MuJoCo's mj_forward() can trigger SIGFPE on certain joint configurations
# (observed intermittently with walk/pace/bound gaits on WSL2).
# Strategy: (1) mask CPU FP exceptions, (2) Python signal handler → exception.

_libm = ctypes.CDLL(ctypes.util.find_library("m"))

# x86_64 Linux fenv.h constants
_FE_INVALID    = 0x01
_FE_DIVBYZERO  = 0x04
_FE_OVERFLOW   = 0x08
_FE_UNDERFLOW  = 0x10
_FE_INEXACT    = 0x20
_FE_ALL_EXCEPT = (_FE_INVALID | _FE_DIVBYZERO | _FE_OVERFLOW |
                  _FE_UNDERFLOW | _FE_INEXACT)


class _MuJoCoFPE(FloatingPointError):
    """Raised when MuJoCo triggers SIGFPE during forward kinematics."""
    pass


def _sigfpe_handler(signum, frame):
    raise _MuJoCoFPE("MuJoCo mj_forward triggered SIGFPE")


def _forward_safe(sim) -> bool:
    """Call sim.forward() — returns False if MuJoCo triggers SIGFPE."""
    prev_fe = _libm.fegetexcept()
    _libm.fedisableexcept(_FE_ALL_EXCEPT)
    prev_sig = signal.signal(signal.SIGFPE, _sigfpe_handler)
    try:
        sim.forward()
        return True
    except _MuJoCoFPE:
        return False
    finally:
        signal.signal(signal.SIGFPE, prev_sig)
        _libm.feenableexcept(prev_fe)

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

        # Cache joint limits for safe IK (prevents MuJoCo numerical issues)
        self._joint_limits = []
        for jname in self._joint_qpos_names:
            jid = sim._joint_ids.get(jname)
            if jid is not None and jid < sim._model.njnt:
                lo = sim._model.jnt_range[jid][0]
                hi = sim._model.jnt_range[jid][1]
                # Add a small margin (1%) to avoid boundary instabilities
                margin = (hi - lo) * 0.01
                self._joint_limits.append((lo + margin, hi - margin))
            else:
                self._joint_limits.append((-np.inf, np.inf))

        # Pre-compute the offset between analytical FK and MuJoCo FK.
        # The MJCF model has a thigh lateral offset [0, ±0.08, 0] that the
        # analytical LegKinematics ignores.  Since this offset lives in the
        # hip body (applied *before* any rotation), it is constant in hip
        # frame for fixed-base simulation.  Set via set_mujoco_offset().
        self._moco_offset = np.zeros(3)

    def set_mujoco_offset(self, offset: np.ndarray):
        """Set the constant hip-frame offset from analytical FK to MuJoCo FK."""
        self._moco_offset = np.asarray(offset, dtype=float)

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
                 damping: float = 0.01) -> Optional[np.ndarray]:
        """Solve IK for a world-frame target foot position.

        Gauss-Newton with damped least squares and analytical Jacobian
        (LegKinematics). Eliminates per-iteration MuJoCo FK calls — only
        one mj_forward at the very end to verify the solution. This avoids
        the intermittent SIGFPE triggered by MuJoCo on certain joint configs.

        Returns joint angles or None if unreachable.
        """
        target = np.asarray(target_world, dtype=float)

        if q0 is None:
            q = self.get_current_joint_angles()
        else:
            q = np.asarray(q0, dtype=float).copy()

        # Transform world target to hip frame for analytical FK/Jacobian
        hip_pos = self.get_hip_frame_position()
        hip_rot = self._sim.get_body_rotation(f"{self._leg}_hip")
        target_hip = hip_rot.T @ (target - hip_pos)

        for iteration in range(max_iter):
            q = self._clamp_q(q)

            # ── FK via analytical model + MuJoCo offset correction ──
            foot_hip, _ = self._kin.forward_kinematics(q)
            foot_hip_corrected = foot_hip + self._moco_offset
            foot_world = hip_pos + hip_rot @ foot_hip_corrected

            error = target - foot_world

            if np.linalg.norm(error) < tol:
                break

            # ── Analytical Jacobian (hip frame) ──
            J_hip = self._kin.jacobian(q)
            J = hip_rot @ J_hip                     # world frame

            # Damped least squares step
            JJT = J @ J.T
            I3 = np.eye(3)
            try:
                dq = J.T @ np.linalg.solve(JJT + damping**2 * I3, error)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(J).dot(error)

            # Fixed step (no line search — damping prevents divergence)
            q = q + dq

        # ── Final verification via MuJoCo FK (single mj_forward) ──
        q = self._clamp_q(q)
        for jname, angle in zip(self._joint_qpos_names, q):
            self._sim.set_qpos(jname, angle)
        if not _forward_safe(self._sim):
            return None
        foot_final = self.get_foot_position()
        if not np.all(np.isfinite(foot_final)):
            return None
        # Verify against MuJoCo FK.  Floating-base models tolerate slightly
        # larger errors because the analytical FK offset is only approximate
        # when the body tilts during locomotion.
        verify_tol = tol * 100  # 1e-3
        max_tol = 0.05          # 5 cm — floating base needs more slack
        if np.linalg.norm(target - foot_final) < verify_tol:
            return q
        if np.linalg.norm(target - foot_final) < max_tol:
            return q
        return None

    def _clamp_q(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to valid ranges (with 1% margin)."""
        q = np.asarray(q, dtype=float)
        for i in range(3):
            lo, hi = self._joint_limits[i]
            q[i] = np.clip(q[i], lo, hi)
        return q

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
