"""Cartesian foot trajectory generators for robotic dog simulation.

Provides various trajectory types in 3D Cartesian space that the robot foot
tip should follow: linear, circular, sinusoidal, Bezier, and step trajectories.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Trajectory(ABC):
    """Abstract base class for Cartesian trajectories."""

    @abstractmethod
    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate trajectory at time t, returning a 3D position (x, y, z)."""
        ...

    @abstractmethod
    def duration(self) -> float:
        """Return the total duration of the trajectory."""
        ...

    def sample(self, dt: float) -> np.ndarray:
        """Sample trajectory at regular intervals, returning (N, 3) array."""
        T = self.duration()
        n = int(T / dt) + 1
        times = np.linspace(0, T, n)
        return np.array([self.evaluate(t) for t in times])


class LinearTrajectory(Trajectory):
    """Linear interpolation between start and end points."""

    def __init__(self, start: np.ndarray, end: np.ndarray, T: float = 1.0):
        self._start = np.asarray(start, dtype=float)
        self._end = np.asarray(end, dtype=float)
        self._T = T

    def evaluate(self, t: float) -> np.ndarray:
        t = max(0, min(t, self._T))
        s = t / self._T
        return self._start + s * (self._end - self._start)

    def duration(self) -> float:
        return self._T


class CircleTrajectory(Trajectory):
    """Circular trajectory in a specified plane."""

    def __init__(self, center: np.ndarray, radius: float,
                 axis: str = "z", T: float = 1.0, phase: float = 0.0):
        """Create a circular trajectory.

        Args:
            center: 3D center of the circle.
            radius: Radius of the circle.
            axis: Normal axis of the circle plane ('x', 'y', or 'z').
            T: Duration for one full revolution.
            phase: Initial phase offset in radians.
        """
        self._center = np.asarray(center, dtype=float)
        self._radius = radius
        self._axis = axis.lower()
        self._T = T
        self._phase = phase

    def evaluate(self, t: float) -> np.ndarray:
        t = max(0, min(t, self._T))
        angle = 2 * np.pi * t / self._T + self._phase
        if self._axis == "z":
            return self._center + np.array([
                self._radius * np.cos(angle),
                self._radius * np.sin(angle),
                0.0,
            ])
        elif self._axis == "y":
            return self._center + np.array([
                self._radius * np.cos(angle),
                0.0,
                self._radius * np.sin(angle),
            ])
        elif self._axis == "x":
            return self._center + np.array([
                0.0,
                self._radius * np.cos(angle),
                self._radius * np.sin(angle),
            ])
        else:
            raise ValueError(f"Unknown axis: {self._axis}")

    def duration(self) -> float:
        return self._T


class SinusoidalTrajectory(Trajectory):
    """Sinusoidal oscillation trajectory."""

    def __init__(self, center: np.ndarray, amplitude: np.ndarray,
                 frequency: float = 1.0, T: float = 1.0,
                 phase: Optional[np.ndarray] = None):
        """Create a sinusoidal trajectory.

        Args:
            center: Midpoint of oscillation.
            amplitude: Amplitude along each axis [ax, ay, az].
            frequency: Oscillation frequency in Hz.
            T: Total duration.
            phase: Phase offset per axis.
        """
        self._center = np.asarray(center, dtype=float)
        self._amplitude = np.asarray(amplitude, dtype=float)
        self._frequency = frequency
        self._T = T
        self._phase = np.asarray(phase if phase is not None else [0, 0, 0])

    def evaluate(self, t: float) -> np.ndarray:
        t = max(0, min(t, self._T))
        omega = 2 * np.pi * self._frequency
        return self._center + self._amplitude * np.sin(omega * t + self._phase)

    def duration(self) -> float:
        return self._T


class LissajousTrajectory(Trajectory):
    """Lissajous curve trajectory."""

    def __init__(self, center: np.ndarray, amplitude: np.ndarray,
                 frequencies: tuple = (1, 2, 1), T: float = 1.0,
                 phase: Optional[np.ndarray] = None):
        self._center = np.asarray(center, dtype=float)
        self._amplitude = np.asarray(amplitude, dtype=float)
        self._freqs = frequencies
        self._T = T
        self._phase = np.asarray(phase if phase is not None else [0, 0, 0])

    def evaluate(self, t: float) -> np.ndarray:
        t = max(0, min(t, self._T))
        omega = 2 * np.pi * t / self._T
        return self._center + self._amplitude * np.sin(
            np.array(self._freqs) * omega + self._phase
        )

    def duration(self) -> float:
        return self._T


# ═══════════════════════════════════════════════════════════════════════════════
# Quintic (5th-order) polynomial trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class QuinticTrajectory1D:
    """5th-order (quintic) polynomial trajectory with C² continuity at boundaries.

    x(t)  = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
    v(t)  = a₁ + 2a₂t + 3a₃t² + 4a₄t³ + 5a₅t⁴
    a(t)  = 2a₂ + 6a₃t + 12a₄t² + 20a₅t³

    The 6 coefficients are solved from 6 boundary conditions:
      x(0)=x0, v(0)=v0, a(0)=a0
      x(T)=xT, v(T)=vT, a(T)=aT

    This is a standalone scalar class — it does NOT inherit from Trajectory
    (which returns ndarray). Use QuinticTrajectory3D for a 3D vector variant.
    """

    def __init__(self, T: float, x0: float, xT: float,
                 v0: float = 0.0, a0: float = 0.0,
                 vT: float = 0.0, aT: float = 0.0):
        if T <= 0:
            raise ValueError(f"Duration T must be positive, got {T}")
        self._T = T
        self._x0, self._v0, self._a0 = x0, v0, a0
        self._xT, self._vT, self._aT = xT, vT, aT

        # Solve coefficients: a0, a1, a2 from initial conditions directly
        a0 = x0
        a1 = v0
        a2 = a0 / 2.0

        # Remaining 3 coefficients [a3, a4, a5] from terminal conditions
        # M @ [a3, a4, a5]^T = rhs
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        M = np.array([
            [T3,       T4,       T5],
            [3 * T2,   4 * T3,   5 * T4],
            [6 * T,    12 * T2,  20 * T3],
        ])
        rhs = np.array([
            xT - a0 - a1 * T - a2 * T2,
            vT - a1 - 2 * a2 * T,
            aT - 2 * a2,
        ])

        self._coeffs = np.zeros(6)
        self._coeffs[0] = a0
        self._coeffs[1] = a1
        self._coeffs[2] = a2
        self._coeffs[3:] = np.linalg.solve(M, rhs)

    @property
    def coefficients(self) -> np.ndarray:
        """Return the 6 polynomial coefficients [a₀, …, a₅]."""
        return self._coeffs.copy()

    def evaluate(self, t: float) -> float:
        """Evaluate position at normalised time t ∈ [0, T]."""
        t = max(0.0, min(t, self._T))
        return np.polyval(self._coeffs[::-1], t)

    def evaluate_vel(self, t: float) -> float:
        """Evaluate velocity at normalised time t ∈ [0, T]."""
        t = max(0.0, min(t, self._T))
        # derivative coefficients: [5a₅, 4a₄, 3a₃, 2a₂, a₁]
        d_coeffs = np.array([
            5 * self._coeffs[5],
            4 * self._coeffs[4],
            3 * self._coeffs[3],
            2 * self._coeffs[2],
            self._coeffs[1],
        ])
        return np.polyval(d_coeffs, t)

    def evaluate_acc(self, t: float) -> float:
        """Evaluate acceleration at normalised time t ∈ [0, T]."""
        t = max(0.0, min(t, self._T))
        # second derivative coefficients: [20a₅, 12a₄, 6a₃, 2a₂]
        dd_coeffs = np.array([
            20 * self._coeffs[5],
            12 * self._coeffs[4],
            6 * self._coeffs[3],
            2 * self._coeffs[2],
        ])
        return np.polyval(dd_coeffs, t)

    def duration(self) -> float:
        return self._T


class QuinticTrajectory3D(Trajectory):
    """3D quintic trajectory composing three per-axis QuinticTrajectory1D instances.

    Inherits from Trajectory so it can be used wherever the abstract base is expected.
    """

    def __init__(self, T: float,
                 x0: np.ndarray, xT: np.ndarray,
                 v0: np.ndarray = None, a0: np.ndarray = None,
                 vT: np.ndarray = None, aT: np.ndarray = None):
        self._T = T
        x0 = np.asarray(x0, dtype=float)
        xT = np.asarray(xT, dtype=float)
        v0 = np.asarray(v0 if v0 is not None else [0, 0, 0], dtype=float)
        a0 = np.asarray(a0 if a0 is not None else [0, 0, 0], dtype=float)
        vT = np.asarray(vT if vT is not None else [0, 0, 0], dtype=float)
        aT = np.asarray(aT if aT is not None else [0, 0, 0], dtype=float)

        self._trajs = [
            QuinticTrajectory1D(T, x0[i], xT[i], v0[i], a0[i], vT[i], aT[i])
            for i in range(3)
        ]

    def evaluate(self, t: float) -> np.ndarray:
        return np.array([traj.evaluate(t) for traj in self._trajs])

    def evaluate_vel(self, t: float) -> np.ndarray:
        return np.array([traj.evaluate_vel(t) for traj in self._trajs])

    def evaluate_acc(self, t: float) -> np.ndarray:
        return np.array([traj.evaluate_acc(t) for traj in self._trajs])

    def duration(self) -> float:
        return self._T
