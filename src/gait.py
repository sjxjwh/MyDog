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

    Usage:
        scheduler = GaitScheduler(GaitType.TROT, params)
        planner = FootTrajectoryPlanner(scheduler, neutral_foot)
        target = planner.get_target_hip("FR", t=0.3)  # → [x, y, z] in hip frame
    """

    def __init__(self, scheduler: GaitScheduler,
                 neutral_foot: Dict[str, np.ndarray]):
        """Initialize the trajectory planner.

        Args:
            scheduler: GaitScheduler managing phase and leg state.
            neutral_foot: Dict mapping leg name to neutral foot position
                         in hip frame (e.g., from FK at home configuration).
        """
        self._scheduler = scheduler
        self._neutral = {leg: np.asarray(pos, dtype=float)
                         for leg, pos in neutral_foot.items()}

    @property
    def scheduler(self) -> GaitScheduler:
        return self._scheduler

    @property
    def params(self) -> GaitParams:
        return self._scheduler.params

    def get_neutral(self, leg: str) -> np.ndarray:
        """Return the neutral foot position for a leg (hip frame)."""
        return self._neutral[leg].copy()

    def get_target_hip(self, leg: str, t: float) -> np.ndarray:
        """Compute desired foot position in hip frame at time t.

        Args:
            leg: Leg identifier ('FR', 'FL', 'RR', 'RL').
            t: Current simulation time in seconds.

        Returns:
            3D position [x, y, z] in hip frame.
        """
        state = self._scheduler.leg_state(leg, t)
        s = self._scheduler.phase_in_state(leg, t)  # progress ∈ [0, 1]
        n = self._neutral[leg]  # neutral foot position [nx, ny, nz]
        L = self.params.step_length
        H = self.params.step_height

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
