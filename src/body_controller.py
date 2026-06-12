"""Full-body controller coordinating all four legs for quadruped locomotion.

Ties together GaitScheduler, FootTrajectoryPlanner, and per-leg IKFootController
to produce coordinated walking gaits on the Unitree Go1 robot.
"""

from typing import Dict, Optional

import numpy as np

from .controller import IKFootController, create_go1_leg_kinematics
from .gait import (
    ALL_LEGS,
    GaitParams,
    GaitScheduler,
    GaitType,
    FootTrajectoryPlanner,
)
from .simulator import MuJoCoSim


# Home joint angles for Go1 (standing pose, all legs symmetric).
HOME_ANGLES = np.array([0.0, 0.9, -1.8])


class BodyController:
    """Coordinates all four legs for gait-based locomotion.

    For each simulation step:
      1. Compute foot target in hip frame via FootTrajectoryPlanner.
      2. Transform to world frame using hip body position.
      3. Solve IK for each leg independently via IKFootController.
      4. Advance physics by one step.

    The controller assumes a FIXED-BASE simulation where the trunk does not
    move. Hip positions are constant throughout the simulation.

    Usage:
        sim = MuJoCoSim("model/go1_fixed.xml")
        bc = BodyController(sim, GaitType.TROT, GaitParams(step_length=0.08))
        for step in range(1000):
            bc.control(t=step * dt)
            bc.step()
    """

    def __init__(self, sim: MuJoCoSim, gait_type: GaitType,
                 params: Optional[GaitParams] = None,
                 warm_up: float = 0.2, floating: bool = False):
        """Initialize the body controller.

        Sets all leg joints to home configuration, computes neutral foot
        positions via MuJoCo FK, and creates per-leg IK controllers.

        Args:
            sim: MuJoCo simulation wrapper.
            gait_type: Type of gait (TROT, WALK, PACE, BOUND).
            params: Gait parameters. Uses defaults if None.
            warm_up: Duration (seconds) to ramp step parameters from 0
                     to target, preventing large initial IK jumps.
            floating: If True, the model has a freejoint (body can move).
                      Hip positions are updated each control step.
        """
        self._sim = sim
        self._params = params or GaitParams()
        self._warm_up = warm_up
        self._floating = floating

        # ── Gait scheduler & trajectory planner ──
        self._scheduler = GaitScheduler(gait_type, self._params)

        # ── Per-leg IK controllers ──
        self._controllers: Dict[str, IKFootController] = {}
        for leg in ALL_LEGS:
            kin = create_go1_leg_kinematics(leg)
            self._controllers[leg] = IKFootController(sim, leg, kin)

        # ── Set home angles and compute neutral foot positions ──
        self._neutral_foot: Dict[str, np.ndarray] = {}
        self._hip_positions: Dict[str, np.ndarray] = {}
        self._hip_rotations: Dict[str, np.ndarray] = {}

        for leg in ALL_LEGS:
            ctrl = self._controllers[leg]
            for jname, angle in zip(ctrl.joint_names, HOME_ANGLES):
                sim.set_qpos(jname, angle)

        sim.forward()

        # ── Floating base: lower body so feet touch the ground ──
        if self._floating:
            # Find the lowest foot z in world frame
            foot_z_min = float("inf")
            for leg in ALL_LEGS:
                fz = self._controllers[leg].get_foot_position()[2]
                foot_z_min = min(foot_z_min, fz)
            # Adjust freejoint z so lowest foot is exactly on ground (z=0)
            sim._data.qpos[2] -= foot_z_min
            sim.forward()

        # ── Neutral foot positions (hip frame) and MuJoCo offset ──
        for leg in ALL_LEGS:
            ctrl = self._controllers[leg]
            foot_world = ctrl.get_foot_position()
            hip_pos = ctrl.get_hip_frame_position()
            hip_rot = sim.get_body_rotation(f"{leg}_hip")

            # Foot position in hip frame: R_hip^T @ (foot_world - hip_pos)
            foot_hip = hip_rot.T @ (foot_world - hip_pos)
            self._neutral_foot[leg] = foot_hip
            self._hip_positions[leg] = hip_pos
            self._hip_rotations[leg] = hip_rot

            # MuJoCo-vs-analytical FK offset (constant in hip frame)
            kin = create_go1_leg_kinematics(leg)
            foot_analytic_hip, _ = kin.forward_kinematics(HOME_ANGLES)
            ctrl.set_mujoco_offset(foot_hip - foot_analytic_hip)

        self._planner = FootTrajectoryPlanner(self._scheduler,
                                               self._neutral_foot,
                                               warm_up=self._warm_up)

        # ── Floating-base state tracking ──
        self._stance_anchor: Dict[str, np.ndarray] = {}   # world XY for stance
        self._last_gait_state: Dict[str, str] = {leg: "stance" for leg in ALL_LEGS}

        # ── Recording state ──
        self._recording = False
        self._record_data: Dict = {}
        self._ik_failures: Dict[str, int] = {leg: 0 for leg in ALL_LEGS}

    # ── Properties ──────────────────────────────────────────────

    @property
    def scheduler(self) -> GaitScheduler:
        return self._scheduler

    @property
    def planner(self) -> FootTrajectoryPlanner:
        return self._planner

    @property
    def params(self) -> GaitParams:
        return self._params

    @property
    def controllers(self) -> Dict[str, IKFootController]:
        return self._controllers

    @property
    def neutral_foot(self) -> Dict[str, np.ndarray]:
        return self._neutral_foot

    @property
    def hip_positions(self) -> Dict[str, np.ndarray]:
        return self._hip_positions

    @property
    def ik_failures(self) -> Dict[str, int]:
        return self._ik_failures

    # ── Core control loop ──────────────────────────────────────

    def stand(self, t: float):
        """Standing mode: hold home angles, no trajectory tracking.

        Joint actuators are set directly to home angles, locking the legs
        in place.  Much more stable than IK-based standing because it
        eliminates the foot-sliding feedback loop.
        """
        for leg in ALL_LEGS:
            ctrl = self._controllers[leg]
            for cname, angle in zip(ctrl._ctrl_names, HOME_ANGLES):
                self._sim.set_joint_ctrl(cname, angle)

    def control(self, t: float):
        """Compute and apply controls for all four legs at time t.

        For each leg:
          1. Compute desired foot position in hip frame via trajectory planner.
          2. Transform to world frame.
          3. Solve IK and set actuator targets.

        For floating-base models, hip positions (and rotations) are refreshed
        from the current simulation state each step.

        Args:
            t: Current simulation time in seconds.
        """
        # ── Floating base: refresh hip transforms ──
        if self._floating:
            self._update_hip_transforms()

        for leg in ALL_LEGS:
            if self._floating:
                # ── Floating base: stance anchoring + hip-frame swing ──
                # Stance feet are ANCHORED at their world XY touch-down position
                # to prevent sliding.  Swing feet follow the planned hip-frame
                # trajectory.  This is open-loop (no body velocity feedback),
                # so the body stays roughly in place — proper locomotion requires
                # force control (see force_controller.py).
                gait_state = self._scheduler.leg_state(leg, t)
                prev_state = self._last_gait_state[leg]

                if gait_state == "stance":
                    if prev_state == "swing" or leg not in self._stance_anchor:
                        # Transition swing→stance: anchor at current world XY
                        foot_now = self._controllers[leg].get_foot_position()
                        self._stance_anchor[leg] = foot_now[:2].copy()
                    target_world_xy = self._stance_anchor[leg]
                    target_world_z = 0.0  # on ground
                else:
                    # Swing: hip-frame XY → world, Z lift from ground
                    target_hip_xy = self._planner.get_target_hip_xy(leg, t)
                    target_world_z = self._planner.get_target_world_z(leg, t)
                    hip_pos = self._hip_positions[leg]
                    hip_rot = self._hip_rotations[leg]
                    target_world_xy = hip_pos[:2] + (hip_rot[:2, :2] @ target_hip_xy)

                self._last_gait_state[leg] = gait_state
                target_world = np.array([target_world_xy[0], target_world_xy[1], target_world_z])
            else:
                # Fixed base: full target in hip frame → world frame
                target_hip = self._planner.get_target_hip(leg, t)
                target_world = (self._hip_positions[leg]
                                + self._hip_rotations[leg] @ target_hip)

            q_target = self._controllers[leg].control(target_world)
            if q_target is None:
                self._ik_failures[leg] += 1

        if self._recording:
            self._record_step(t)

    def step(self):
        """Advance the simulation by one physics step."""
        self._sim.step()

    def settle(self, duration: float = 0.3, dt: float = 0.002):
        """Let body settle on ground with position control.

        Two-phase approach:
        1. Position-settle at home angles — lets PD forces reach equilibrium.
        2. Adjust freejoint z so feet touch ground, then settle again.
        """
        n_steps = int(duration / dt)
        n_half = n_steps // 2

        # Phase 1: position settle
        for _ in range(n_half):
            for leg in ALL_LEGS:
                ctrl = self._controllers[leg]
                for cname, angle in zip(ctrl._ctrl_names, HOME_ANGLES):
                    self._sim.set_joint_ctrl(cname, angle)
            self._sim.step()

        # Phase 2: lower body so feet touch ground
        self._sim.forward()
        foot_z_min = min(
            self._controllers[leg].get_foot_position()[2]
            for leg in ALL_LEGS
        )
        if foot_z_min > 0.002:  # feet are above ground
            self._sim._data.qpos[2] -= foot_z_min
            self._sim.forward()

        # Phase 3: position settle again at new height
        for _ in range(n_half):
            for leg in ALL_LEGS:
                ctrl = self._controllers[leg]
                for cname, angle in zip(ctrl._ctrl_names, HOME_ANGLES):
                    self._sim.set_joint_ctrl(cname, angle)
            self._sim.step()

        self._sim.forward()
        self._update_hip_transforms()

    def _update_hip_transforms(self):
        """Refresh hip world positions and rotations from MuJoCo state.

        Called every control step for floating-base models where the body
        can translate and rotate during simulation.
        """
        for leg in ALL_LEGS:
            hip_body = f"{leg}_hip"
            self._hip_positions[leg] = self._sim.get_body_position(hip_body)
            self._hip_rotations[leg] = self._sim.get_body_rotation(hip_body)

    # ── State queries ──────────────────────────────────────────

    def get_foot_position(self, leg: str) -> np.ndarray:
        """Get current world-frame foot position for a leg."""
        return self._controllers[leg].get_foot_position()

    def get_joint_angles(self, leg: str) -> np.ndarray:
        """Get current joint angles for a leg."""
        return self._controllers[leg].get_current_joint_angles()

    def get_all_foot_positions(self) -> Dict[str, np.ndarray]:
        """Get world-frame foot positions for all legs."""
        return {leg: self.get_foot_position(leg) for leg in ALL_LEGS}

    def get_all_joint_angles(self) -> Dict[str, np.ndarray]:
        """Get joint angles for all legs."""
        return {leg: self.get_joint_angles(leg) for leg in ALL_LEGS}

    def get_leg_state(self, leg: str, t: float) -> str:
        """Get current gait state for a leg."""
        return self._scheduler.leg_state(leg, t)

    # ── Data recording ─────────────────────────────────────────

    def start_recording(self, total_steps: int):
        """Begin recording simulation data for later analysis.

        Args:
            total_steps: Expected number of simulation steps.
        """
        self._recording = True
        n_legs = len(ALL_LEGS)
        self._record_data = {
            "time": np.zeros(total_steps),
            "target": {leg: np.zeros((total_steps, 3)) for leg in ALL_LEGS},
            "actual": {leg: np.zeros((total_steps, 3)) for leg in ALL_LEGS},
            "joint_targets": {leg: np.zeros((total_steps, 3)) for leg in ALL_LEGS},
            "joint_actual": {leg: np.zeros((total_steps, 3)) for leg in ALL_LEGS},
        }
        self._record_step_idx = 0
        self._ik_failures = {leg: 0 for leg in ALL_LEGS}

    def _record_step(self, t: float):
        """Record data for the current simulation step."""
        idx = self._record_step_idx
        self._record_data["time"][idx] = t
        for leg in ALL_LEGS:
            if self._floating:
                # For floating mode, use actual foot position as reference
                # (target is computed dynamically in control())
                target_world = self._controllers[leg].get_foot_position()
                target_world[2] = 0.0  # ground level
            else:
                target_hip = self._planner.get_target_hip(leg, t)
                target_world = (self._hip_positions[leg]
                                + self._hip_rotations[leg] @ target_hip)
            actual_world = self.get_foot_position(leg)
            joint_actual = self.get_joint_angles(leg)
            # Get last IK target by reading ctrl values
            joint_target = np.zeros(3)
            for j, jname in enumerate(self._controllers[leg].joint_names):
                cname = self._controllers[leg]._ctrl_names[j]
                try:
                    joint_target[j] = self._sim._data.ctrl[
                        self._sim._actuator_ids[cname]
                    ]
                except KeyError:
                    joint_target[j] = joint_actual[j]

            self._record_data["target"][leg][idx] = target_world
            self._record_data["actual"][leg][idx] = actual_world
            self._record_data["joint_targets"][leg][idx] = joint_target
            self._record_data["joint_actual"][leg][idx] = joint_actual

        self._record_step_idx += 1

    def get_recorded_data(self) -> Dict:
        """Return recorded simulation data."""
        return self._record_data

    # ── Summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a summary of the controller state."""
        lines = [
            f"BodyController — {self._scheduler.gait_type.value} gait",
            f"  T_cycle={self.params.T_cycle:.2f}s, "
            f"duty={self.params.duty_factor:.2f}, "
            f"step_len={self.params.step_length:.3f}m, "
            f"step_h={self.params.step_height:.3f}m",
        ]
        for leg in ALL_LEGS:
            foot = self.get_foot_position(leg)
            state = self._scheduler.leg_state(leg, 0.0)
            lines.append(
                f"  {leg}: foot=[{foot[0]:.4f}, {foot[1]:.4f}, {foot[2]:.4f}] "
                f"(neutral_hip=[{self._neutral_foot[leg][0]:.4f}, "
                f"{self._neutral_foot[leg][1]:.4f}, "
                f"{self._neutral_foot[leg][2]:.4f}])"
            )
        return "\n".join(lines)
