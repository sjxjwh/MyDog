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
