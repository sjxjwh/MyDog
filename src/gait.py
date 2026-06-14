"""Gait scheduler and foot trajectory planning for quadruped locomotion.

Provides gait phase management and Cartesian foot trajectory generation
for coordinated multi-leg locomotion patterns (trot, walk, pace, bound).

Architecture:
  GaitScheduler      — decides which leg is in stance/swing at time t
  FootTrajectoryPlanner — generates hip-frame foot targets per leg
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np


class GaitType(Enum):
    """Supported quadruped gait patterns."""
    TROT = "trot"    # Diagonal pairs synchronized (FR+RL, FL+RR)
    WALK = "walk"    # Wave gait — at least 3 feet on ground at all times
    PACE = "pace"    # Lateral pairs synchronized (FR+FL, RR+RL)
    BOUND = "bound"  # Front pair and rear pair synchronized


# Phase offsets [0, 1) for each leg under each gait type.
# These define the relative timing of each leg within a gait cycle.
GAIT_OFFSETS: Dict[GaitType, Dict[str, float]] = {
    GaitType.TROT:  {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0},
    GaitType.WALK:  {"FR": 0.0, "FL": 0.75, "RR": 0.5, "RL": 0.25},
    GaitType.PACE:  {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5},
    GaitType.BOUND: {"FR": 0.0, "FL": 0.0, "RR": 0.0, "RL": 0.0},
}

ALL_LEGS = ["FR", "FL", "RR", "RL"]


@dataclass
class GaitParams:
    """Parameters defining a gait pattern.

    Attributes:
        T_cycle: Gait cycle period in seconds. One full cycle for each leg.
        duty_factor: Fraction of the cycle spent in stance (ground contact).
                     Must be in (0, 1). Higher = more stable, lower = faster.
        step_length: Total x-displacement of the foot per step (meters).
                     The foot moves ± step_length/2 around the neutral point.
        step_height: Maximum foot lift height during swing phase (meters).
    """
    T_cycle: float = 0.5
    duty_factor: float = 0.6
    step_length: float = 0.06
    step_height: float = 0.04

    def __post_init__(self):
        if not 0 < self.duty_factor < 1:
            raise ValueError(f"duty_factor must be in (0, 1), got {self.duty_factor}")
        if self.T_cycle <= 0:
            raise ValueError(f"T_cycle must be positive, got {self.T_cycle}")


class GaitScheduler:
    """Manages gait phase and leg state for a quadruped.

    Each leg's phase φ ∈ [0, 1) advances linearly with time:
        φ(leg, t) = (offset[leg] + t / T_cycle) mod 1

    The leg is in STANCE when φ < duty_factor, and in SWING otherwise.

    Usage:
        scheduler = GaitScheduler(GaitType.TROT, GaitParams(T_cycle=0.5))
        state = scheduler.leg_state("FR", t=1.2)  # → 'stance' or 'swing'
    """

    def __init__(self, gait_type: GaitType, params: GaitParams):
        self._gait_type = gait_type
        self._params = params
        self._offsets = GAIT_OFFSETS[gait_type]

    @property
    def gait_type(self) -> GaitType:
        return self._gait_type

    @property
    def params(self) -> GaitParams:
        return self._params

    @property
    def T_cycle(self) -> float:
        return self._params.T_cycle

    @property
    def duty_factor(self) -> float:
        return self._params.duty_factor

    def phase(self, leg: str, t: float) -> float:
        """Return the normalized gait phase for a leg at time t.

        Args:
            leg: Leg identifier ('FR', 'FL', 'RR', 'RL').
            t: Current simulation time in seconds.

        Returns:
            Phase φ ∈ [0, 1), where 0 marks the start of stance.
        """
        offset = self._offsets[leg]
        return (offset + t / self.T_cycle) % 1.0

    def leg_state(self, leg: str, t: float) -> str:
        """Return the state of a leg at time t.

        Args:
            leg: Leg identifier.
            t: Current simulation time.

        Returns:
            'stance' if the leg is in ground contact, 'swing' otherwise.
        """
        return "stance" if self.phase(leg, t) < self.duty_factor else "swing"

    def phase_in_state(self, leg: str, t: float) -> float:
        """Return normalized progress within the current state [0, 1].

        0 = start of current state, 1 = end of current state.

        Args:
            leg: Leg identifier.
            t: Current simulation time.

        Returns:
            Progress s ∈ [0, 1] within the current stance or swing phase.
        """
        p = self.phase(leg, t)
        if p < self.duty_factor:
            return p / self.duty_factor
        else:
            return (p - self.duty_factor) / (1.0 - self.duty_factor)

    def get_stance_legs(self, t: float) -> list[str]:
        """Return list of legs currently in stance."""
        return [leg for leg in ALL_LEGS if self.leg_state(leg, t) == "stance"]

    def get_swing_legs(self, t: float) -> list[str]:
        """Return list of legs currently in swing."""
        return [leg for leg in ALL_LEGS if self.leg_state(leg, t) == "swing"]


class FootTrajectoryPlanner:
    """Generates Cartesian foot trajectories in the hip frame for each leg.

    Each leg's foot follows a periodic trajectory synchronized with the gait:
      - STANCE:  Foot moves linearly backward (X decreases) at constant Z.
                 Simulates the body moving forward while foot stays on ground.
      - SWING:   Foot lifts in a sinusoidal arc and swings forward (X increases)
                 to the next foothold.

    The foot position is computed in the HIP FRAME (origin at hip body,
    X forward, Y left, Z up). For fixed-base simulation, the hip frame
    differs from the world frame only by a translation.

    Kinematic velocity control (Raibert-style):
      - step_length adjusted by forward velocity error
      - lateral foot offset adjusted by lateral velocity error
      - differential foot X offset for yaw rate control

    Usage:
        scheduler = GaitScheduler(GaitType.TROT, params)
        planner = FootTrajectoryPlanner(scheduler, neutral_foot)
        target = planner.get_target_hip("FR", t=0.3)  # → [x, y, z] in hip frame
    """

    def __init__(self, scheduler: GaitScheduler,
                 neutral_foot: Dict[str, np.ndarray],
                 warm_up: float = 0.2):
        """Initialize the trajectory planner.

        Args:
            scheduler: GaitScheduler managing phase and leg state.
            neutral_foot: Dict mapping leg name to neutral foot position
                         in hip frame (e.g., from FK at home configuration).
            warm_up: Duration (seconds) to linearly ramp step_length and
                     step_height from 0 to their target values. Prevents
                     large initial IK jumps when a leg starts mid-swing.
        """
        self._scheduler = scheduler
        self._neutral = {leg: np.asarray(pos, dtype=float)
                         for leg, pos in neutral_foot.items()}
        self._warm_up = warm_up
        self.target_vyaw = 0.0  # kinematic yaw rate (rad/s), set by controller

        # ── Kinematic velocity adjustment state ──
        self._step_length_delta: float = 0.0   # forward velocity correction (m)
        self._lateral_offset: float = 0.0      # lateral velocity correction (m)
        self._yaw_offset: float = 0.0          # yaw rate correction (m, per-side)

        # Gains (set by controller, exposed for tuning)
        self.K_kin_vx: float = 0.0
        self.K_kin_vy: float = 0.0
        self.K_kin_wz: float = 0.0

    @property
    def scheduler(self) -> GaitScheduler:
        return self._scheduler

    @property
    def params(self) -> GaitParams:
        return self._scheduler.params

    def get_neutral(self, leg: str) -> np.ndarray:
        """Return the neutral foot position for a leg (hip frame)."""
        return self._neutral[leg].copy()

    # ── Kinematic velocity adjustment interface ────────────────────────

    def set_kinematic_adjustments(self, step_length_delta: float,
                                  lateral_offset: float, yaw_angle: float):
        """Set foot placement adjustments based on velocity errors.

        Called by the controller each control step (~500 Hz).

        Args:
            step_length_delta: Additional step length for forward velocity
                              correction (m).
            lateral_offset: Lateral foot offset for sideways velocity
                           correction (m).
            yaw_angle: Rotation angle (rad) to apply to swing landing
                      position. Positive = CCW body rotation.
        """
        self._step_length_delta = float(step_length_delta)
        self._lateral_offset = float(lateral_offset)
        self._yaw_offset = float(yaw_angle)

    def _effective_neutral(self, leg: str) -> np.ndarray:
        """Return the neutral foot position with kinematic adjustments applied.

        Adjustments:
          - lateral_offset: shifts Y in hip frame (all legs same direction)

        Note: yaw_offset is NOT applied here. It is applied only to the swing
        landing position in _build_swing_trajectories, because the stance
        trajectory and swing start position should not be affected by the
        rotational correction — only the foot's landing point matters.
        """
        n = self._neutral[leg].copy()
        n[1] += self._lateral_offset
        return n

    def _effective_step_length(self, t: float) -> float:
        """Return step_length with velocity correction + warm-up ramp."""
        ramp = min(1.0, t / self._warm_up) if self._warm_up > 0 else 1.0
        return (self.params.step_length + self._step_length_delta) * ramp

    def get_target_hip(self, leg: str, t: float) -> np.ndarray:
        """Compute desired foot position in hip frame at time t.

        Foot placement is adjusted by kinematic velocity feedback:
          - step_length_delta corrects forward velocity
          - lateral_offset corrects lateral velocity
          - yaw_offset creates differential foot placement for turning

        Args:
            leg: Leg identifier ('FR', 'FL', 'RR', 'RL').
            t: Current simulation time in seconds.

        Returns:
            3D position [x, y, z] in hip frame.
        """
        state = self._scheduler.leg_state(leg, t)
        s = self._scheduler.phase_in_state(leg, t)  # progress ∈ [0, 1]
        n = self._effective_neutral(leg)

        # Ramp parameters
        ramp = min(1.0, t / self._warm_up) if self._warm_up > 0 else 1.0
        L = self._effective_step_length(t)
        H = self.params.step_height * ramp

        if state == "stance":
            # Foot moves backward: nx + L/2 → nx - L/2
            x = n[0] + L / 2 - L * s
            y = n[1]
            z = n[2]
        else:
            # Foot swings forward: nx - L/2 → nx + L/2, with lift
            x = n[0] - L / 2 + L * s
            y = n[1]
            z = n[2] + H * np.sin(np.pi * s)

        return np.array([x, y, z])

    def get_target_world(self, leg: str, t: float,
                         hip_positions: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute desired foot position in world frame.

        For fixed-base with no rotation, this is a simple translation:
            target_world = hip_pos + target_hip

        Args:
            leg: Leg identifier.
            t: Current simulation time.
            hip_positions: Dict mapping leg → hip body world position.

        Returns:
            3D position [x, y, z] in world frame.
        """
        target_hip = self.get_target_hip(leg, t)
        hip_pos = np.asarray(hip_positions[leg])
        return hip_pos + target_hip

    def get_target_hip_xy(self, leg: str, t: float) -> np.ndarray:
        """Compute the XY component of the target in hip frame.

        For floating-base, the Z component should be specified in world
        frame (ground-relative), while XY stays in hip frame.
        """
        full = self.get_target_hip(leg, t)
        return full[:2].copy()

    def get_target_world_z(self, leg: str, t: float) -> float:
        """Compute the Z component of the target in WORLD frame.

        For stance: z = 0 (on ground).
        For swing: z = step_height * sin(π * phase_in_swing) (lift).
        """
        state = self._scheduler.leg_state(leg, t)
        s = self._scheduler.phase_in_state(leg, t)
        ramp = min(1.0, t / self._warm_up) if self._warm_up > 0 else 1.0

        if state == "stance":
            return 0.0
        else:
            return ramp * self.params.step_height * np.sin(np.pi * s)


# ═══════════════════════════════════════════════════════════════════════════════
# Quintic foot trajectory planner (C²-continuous swing trajectories)
# ═══════════════════════════════════════════════════════════════════════════════

class QuinticFootTrajectoryPlanner(FootTrajectoryPlanner):
    """Foot trajectory planner using quintic polynomials for C²-continuous swing.

    Inherits stance trajectory (linear backward X) and sinusoidal Z lift from
    FootTrajectoryPlanner.  Replaces the linear-X swing with a quintic polynomial
    that guarantees zero velocity and acceleration at touchdown and liftoff,
    producing smoother foot transitions.

    On each stance→swing transition, three per-axis quintic trajectories are
    built and cached for the duration of the swing phase.
    """

    def __init__(self, scheduler: GaitScheduler,
                 neutral_foot: Dict[str, np.ndarray],
                 warm_up: float = 0.2):
        super().__init__(scheduler, neutral_foot, warm_up)
        self._swing_duration = (1.0 - scheduler.duty_factor) * scheduler.T_cycle
        self._swing_traj: Dict[str, Dict[str, object]] = {
            leg: {"x": None, "y": None, "z_rise": None, "z_fall": None}
            for leg in ALL_LEGS
        }
        self._swing_t0: Dict[str, float] = {leg: 0.0 for leg in ALL_LEGS}
        self._prev_state: Dict[str, str] = {leg: "stance" for leg in ALL_LEGS}

    def _build_swing_trajectories(self, leg: str, t: float,
                                  L: float, n: np.ndarray):
        """Build quintic trajectories for X, Y, Z axes on swing entry.

        X: from end-of-stance position to neutral + L/2 (forward foothold).
        Y: constant at neutral_y → rotated landing (yaw adjustment).
        Z: two-segment quintic: rise (0→T/2) + fall (T/2→T) with C² continuity
           at touchdown and liftoff (v=0, a=0 at both boundaries).
        """
        from .trajectory import QuinticTrajectory1D

        T_sw = self._swing_duration
        if T_sw <= 0:
            T_sw = 0.1  # fallback

        # Landing position in hip frame BEFORE rotation: [nx + L/2, ny]
        x_land0 = n[0] + L / 2.0
        y_land0 = n[1]

        # Yaw: rotate landing position by θ around body z-axis.
        theta = self._yaw_offset
        if abs(theta) > 1e-9:
            c, s = np.cos(theta), np.sin(theta)
            x_end = x_land0 * c - y_land0 * s
            y_end = x_land0 * s + y_land0 * c
        else:
            x_end, y_end = x_land0, y_land0

        x_start = n[0] - L / 2.0
        self._swing_traj[leg]["x"] = QuinticTrajectory1D(
            T_sw, x_start, x_end, v0=0.0, a0=0.0, vT=0.0, aT=0.0,
        )

        self._swing_traj[leg]["y"] = QuinticTrajectory1D(
            T_sw, y_land0, y_end, v0=0.0, a0=0.0, vT=0.0, aT=0.0,
        )

        # ── Z: two-segment quintic for smooth touchdown ──
        # Segment 1 (rise): 0 → T_sw/2, z_neutral → z_neutral + H*ramp
        # Segment 2 (fall): T_sw/2 → T_sw, z_neutral + H*ramp → z_neutral
        # Both with v=0, a=0 at boundaries (C² touchdown and liftoff).
        half_T = T_sw / 2.0
        ramp = min(1.0, t / self._warm_up) if self._warm_up > 0 else 1.0
        H_eff = self.params.step_height * ramp
        z_peak = n[2] + H_eff

        self._swing_traj[leg]["z_rise"] = QuinticTrajectory1D(
            half_T, n[2], z_peak, v0=0.0, a0=0.0, vT=0.0, aT=0.0,
        )
        self._swing_traj[leg]["z_fall"] = QuinticTrajectory1D(
            half_T, z_peak, n[2], v0=0.0, a0=0.0, vT=0.0, aT=0.0,
        )

        self._swing_t0[leg] = t

    def get_target_world_z(self, leg: str, t: float) -> float:
        """Override: quintic Z lift for C² touchdown.

        For stance: 0.0 (on ground).
        For swing: two-segment quintic lift above ground level.
        """
        state = self._scheduler.leg_state(leg, t)
        ramp = min(1.0, t / self._warm_up) if self._warm_up > 0 else 1.0

        if state == "stance":
            return 0.0

        # Swing: use cached quintic trajectory if available, else fall back
        tau = t - self._swing_t0.get(leg, 0.0)
        tau = max(0.0, min(tau, self._swing_duration))

        z_rise = self._swing_traj[leg].get("z_rise")
        if z_rise is not None:
            half_T = self._swing_duration / 2.0
            if tau < half_T:
                z_hip = z_rise.evaluate(tau)
            else:
                z_hip = self._swing_traj[leg]["z_fall"].evaluate(tau - half_T)
            # z_hip includes neutral Z; subtract to get lift height above ground
            z_neutral = self._neutral[leg][2]
            return ramp * (z_hip - z_neutral)
        else:
            # Fallback: sinusoidal (shouldn't happen after first swing buildup)
            s = self._scheduler.phase_in_state(leg, t)
            return ramp * self.params.step_height * np.sin(np.pi * s)

    def get_target_hip(self, leg: str, t: float) -> np.ndarray:
        """Compute desired foot position in hip frame at time t.

        Uses quintic polynomial for swing-phase X and Y (C²-continuous),
        sinusoidal Z lift, and linear stance trajectory.

        Kinematic velocity adjustments (Raibert-style foot placement) are
        inherited from FootTrajectoryPlanner and applied to both stance
        and swing phases.
        """
        state = self._scheduler.leg_state(leg, t)
        s = self._scheduler.phase_in_state(leg, t)
        n = self._effective_neutral(leg)  # includes kinematic adjustments

        # Ramp
        ramp = min(1.0, t / self._warm_up) if self._warm_up > 0 else 1.0
        L = self._effective_step_length(t)
        H = self.params.step_height * ramp

        prev = self._prev_state.get(leg, "stance")

        if state == "stance":
            # Clear cached swing trajectories
            self._swing_traj[leg] = {"x": None, "y": None,
                                      "z_rise": None, "z_fall": None}
            self._prev_state[leg] = "stance"
            # Same linear stance as parent (uses adjusted L and n)
            x = n[0] + L / 2.0 - L * s
            return np.array([x, n[1], n[2]])

        else:  # swing
            # Build trajectories on stance→swing transition (uses adjusted n, L)
            if prev == "stance" or self._swing_traj[leg].get("x") is None:
                self._build_swing_trajectories(leg, t, L, n)

            self._prev_state[leg] = "swing"

            # Evaluate cached quintic trajectories
            tau = t - self._swing_t0[leg]
            tau = max(0.0, min(tau, self._swing_duration))

            x = self._swing_traj[leg]["x"].evaluate(tau)
            y = self._swing_traj[leg]["y"].evaluate(tau)

            # Z: two-segment quintic (rise + fall) for C² touchdown
            half_T = self._swing_duration / 2.0
            if tau < half_T:
                z = self._swing_traj[leg]["z_rise"].evaluate(tau)
            else:
                z = self._swing_traj[leg]["z_fall"].evaluate(tau - half_T)

            return np.array([x, y, z])
