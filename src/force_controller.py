"""MIT-style locomotion controller for quadruped robots.

Architecture (based on MIT Cheetah / Mini Cheetah):
  1. Body PD: desired body wrench from height / roll / pitch / velocity errors
  2. Force distribution: allocate wrench to stance-foot ground reaction forces
  3. Joint torques: τ = Jᵀ · F  (Jacobian transpose)
  4. Swing legs: task-space PD tracking via Jᵀ

All torques are applied via MuJoCo's qfrc_applied, bypassing position actuators.
"""

from typing import Dict

import numpy as np

from .controller import IKFootController, create_go1_leg_kinematics
from .gait import ALL_LEGS, GaitParams, GaitScheduler, FootTrajectoryPlanner
from .simulator import MuJoCoSim


# ── Robot parameters ─────────────────────────────────────────────────────────

GO1_MASS = 12.86       # kg (trunk + 4 legs)
GRAVITY = 9.81
TOTAL_WEIGHT = GO1_MASS * GRAVITY  # ≈ 126 N

# Trunk inertia (approximate, from MJCF diaginertia)
TRUNK_I = np.array([0.07166, 0.06301, 0.01681])  # Ixx, Iyy, Izz


# ═══════════════════════════════════════════════════════════════════════════════
# Per-leg torque control primitive
# ═══════════════════════════════════════════════════════════════════════════════

class LegTorqueController:
    """Applies joint torques from desired foot force via Jacobian transpose."""

    def __init__(self, sim: MuJoCoSim, leg: str):
        kin = create_go1_leg_kinematics(leg)
        self._sim = sim
        self._leg = leg
        self._kin = kin
        self._ik = IKFootController(sim, leg, kin)

        self._dof_addrs: list[int] = []
        for jname in self._ik.joint_names:
            jid = sim._joint_ids.get(jname)
            if jid is not None:
                self._dof_addrs.append(sim._model.jnt_dofadr[jid])

    @property
    def joint_names(self) -> list[str]:
        return self._ik.joint_names

    def get_foot_pos(self) -> np.ndarray:
        return self._ik.get_foot_position()

    def get_foot_vel(self) -> np.ndarray:
        q = self._ik.get_current_joint_angles()
        qvel = np.array([self._sim.get_joint_qvel(jn)
                         for jn in self._ik.joint_names])
        J_hip = self._kin.jacobian(q)
        hip_rot = self._sim.get_body_rotation(f"{self._leg}_hip")
        return hip_rot @ (J_hip @ qvel)

    def get_hip_pos(self) -> np.ndarray:
        return self._ik.get_hip_frame_position()

    def get_hip_rot(self) -> np.ndarray:
        return self._sim.get_body_rotation(f"{self._leg}_hip")

    def jacobian_world(self, q: np.ndarray) -> np.ndarray:
        """3×3 Jacobian: q̇ → foot velocity in WORLD frame."""
        return self.get_hip_rot() @ self._kin.jacobian(q)

    def apply_force(self, F_world: np.ndarray):
        """Apply torque τ = Jᵀ · F to achieve foot force F in world frame."""
        q = self._ik.get_current_joint_angles()
        J = self.jacobian_world(q)
        tau = J.T @ np.asarray(F_world)
        # Clamp torques to prevent numerical instability in MuJoCo
        tau = np.clip(tau, -23.0, 23.0)
        for addr, t in zip(self._dof_addrs, tau):
            self._sim._data.qfrc_applied[addr] = t

    def apply_impedance(self, target_world: np.ndarray,
                        Kp: np.ndarray, Kd: np.ndarray,
                        ff_force: np.ndarray = None,
                        max_force: float = 150.0):
        """Impedance: F = ff + Kp*(xᵈ−x) + Kd*(vᵈ−v), then τ = Jᵀ·F."""
        x = self.get_foot_pos()
        v = self.get_foot_vel()
        F = Kp * (target_world - x) - Kd * v  # vᵈ = 0
        if ff_force is not None:
            F += ff_force
        # Limit force magnitude to prevent numerical instability
        f_norm = np.linalg.norm(F)
        if f_norm > max_force:
            F *= max_force / f_norm
        self.apply_force(F)

    def clear(self):
        for addr in self._dof_addrs:
            self._sim._data.qfrc_applied[addr] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MIT Body Controller
# ═══════════════════════════════════════════════════════════════════════════════

class MITBodyController:
    """MIT-style locomotion controller.

    Body PD computes a desired wrench (force + torque) to track height,
    roll, pitch, and velocity.  The wrench is distributed to stance feet
    as ground reaction forces.  Joint torques are computed via Jᵀ.
    """

    def __init__(self, sim: MuJoCoSim, gait_type,
                 params: GaitParams = None, warm_up: float = 0.3):
        self._sim = sim
        self._params = params or GaitParams()
        self._warm_up = warm_up

        self._scheduler = GaitScheduler(gait_type, self._params)
        self._ctrls = {leg: LegTorqueController(sim, leg) for leg in ALL_LEGS}

        # ── Set home angles and lower body to ground ──
        HOME = np.array([0.0, 0.9, -1.8])
        for leg in ALL_LEGS:
            for jn, a in zip(self._ctrls[leg].joint_names, HOME):
                sim.set_qpos(jn, a)
        sim.forward()
        fz_min = min(self._ctrls[leg].get_foot_pos()[2] for leg in ALL_LEGS)
        sim._data.qpos[2] -= fz_min  # feet at z=0
        sim.forward()

        # ── Neutral foot positions (hip frame) ──
        self._neutral: Dict[str, np.ndarray] = {}
        self._hip_pos: Dict[str, np.ndarray] = {}
        for leg in ALL_LEGS:
            c = self._ctrls[leg]
            f = c.get_foot_pos()
            h = c.get_hip_pos()
            R = c.get_hip_rot()
            self._neutral[leg] = R.T @ (f - h)
            self._hip_pos[leg] = h

        self._planner = FootTrajectoryPlanner(self._scheduler, self._neutral,
                                               warm_up=warm_up)

        # Disable position actuators
        sim._data.ctrl[:] = 0.0

        # ── Body PD gains ──
        self.Kp_z    = 200.0    # height (N/m)
        self.Kd_z    = 40.0     # vertical damping (N·s/m)
        self.Kp_roll = 30.0     # roll (Nm/rad) — low to avoid oscillation
        self.Kd_roll = 10.0     # roll damping
        self.Kp_pitch = 30.0    # pitch
        self.Kd_pitch = 10.0
        self.Kp_vx   = 500.0    # forward velocity (N / (m/s))
        self.Kd_vy   = 50.0     # lateral velocity damping (N / (m/s))
        self.Kd_yaw  = 30.0     # yaw rate damping (Nm / (rad/s))
        self.target_vx = 0.3    # desired forward speed (m/s)

        # Stance / swing gains
        self._stance_Kp = np.array([150.0, 150.0, 500.0])
        self._stance_Kd = np.array([10.0,  10.0,  20.0])
        self._swing_Kp  = np.array([400.0, 400.0, 400.0])
        self._swing_Kd  = np.array([15.0,  15.0,  15.0])

    # ── Body state ────────────────────────────────────────────────────────

    def _body_state(self):
        """Return (pos, quat, vel, angvel) of the trunk in world frame."""
        qpos = self._sim._data.qpos
        qvel = self._sim._data.qvel
        pos = qpos[0:3].copy()               # x, y, z
        quat = qpos[3:7].copy()              # w, x, y, z
        vel = qvel[0:3].copy()               # linear velocity
        angvel = qvel[3:6].copy()            # angular velocity
        # quat → roll, pitch, yaw
        roll, pitch, yaw = self._quat_to_rpy(quat)
        return pos, quat, vel, angvel, roll, pitch, yaw

    @staticmethod
    def _quat_to_rpy(quat):
        """Convert quaternion [w, x, y, z] to roll, pitch, yaw (radians)."""
        w, x, y, z = quat
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    # ── Settle ────────────────────────────────────────────────────────────

    def settle(self, duration: float = 0.5, dt: float = 0.002):
        """Let body stabilize using MuJoCo's built-in position actuators."""
        n = int(duration / dt)
        HOME = np.array([0.0, 0.9, -1.8])

        for _ in range(n):
            for leg in ALL_LEGS:
                self._ctrls[leg].clear()
                for cname, angle in zip(self._ctrls[leg]._ik._ctrl_names, HOME):
                    self._sim.set_joint_ctrl(cname, angle)
            self._sim.step()
        self._sim.forward()

    # ── Main control loop ─────────────────────────────────────────────────

    def control(self, t: float):
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()

        # ── Body PD: desired wrench (force + torque on trunk) ──
        target_height = 0.28
        target_vx = self.target_vx

        # Vertical force (world Z).  Negative = push foot DOWN into ground.
        # Base: -weight/n supports body.  PD: if body too high, reduce support
        # (add positive → less negative); if too low, increase support.
        Fz = -TOTAL_WEIGHT \
             + self.Kp_z * (pos[2] - target_height) \
             - self.Kd_z * vel[2]

        # Roll / pitch torque (disabled for initial tuning)
        Mx = 0.0
        My = 0.0

        # Yaw torque: damp angular velocity around Z (prevent drifting yaw)
        Mz = -self.Kd_yaw * angvel[2]

        # Forward force: velocity tracking.
        # To move body forward, foot pushes BACKWARD on ground
        # → Fx at foot is NEGATIVE (backward).  Reaction pushes body forward.
        Fx = -self.Kp_vx * (target_vx - vel[0])

        # Lateral force: damp sideways velocity
        Fy = -self.Kd_vy * vel[1]

        # ── Force distribution to stance feet ──
        stance = self._scheduler.get_stance_legs(t)
        n = max(1, len(stance))

        # Build contact matrix: each stance leg contributes to body wrench
        # f_i = [fx_i, fy_i, fz_i] at foot position p_i.
        # Body wrench = Σ [f_i; (p_i − com) × f_i]
        com = pos.copy()
        F_alloc = np.zeros(3)  # [Fx, Fy, Fz] distributed
        M_alloc = np.zeros(3)  # [Mx, My, Mz] distributed

        # Simple equal distribution for Fz + height PD, then handle roll/pitch
        foot_forces: Dict[str, np.ndarray] = {}

        hip_positions = {}
        for leg in stance:
            hip_positions[leg] = self._ctrls[leg].get_hip_pos()

        # Compute foot positions and COM
        stance_feet = []
        for leg in stance:
            p = self._ctrls[leg].get_foot_pos()
            stance_feet.append((leg, p))

        if len(stance_feet) == 0:
            return

        # ── Simple QP-free allocation ──
        # Each stance leg gets: 1/n of Fz, 1/n of Fx
        # Roll/pitch moments distributed to left/right and front/rear legs
        for leg, p_foot in stance_feet:
            r = p_foot - com  # vector from COM to foot
            f = np.zeros(3)
            f[0] = Fx / n                      # forward push
            f[1] = Fy / n                      # lateral damping
            f[2] = Fz / n                      # vertical support
            foot_forces[leg] = f

        # Roll moment → differential vertical force on left vs right
        # Left legs: FL, RL (y > 0). Right legs: FR, RR (y < 0).
        left_legs  = [l for l in stance if l[1] == 'L']
        right_legs = [l for l in stance if l[1] == 'R']
        if left_legs and right_legs:
            # Track width ~0.094m
            width = 0.094
            dfz_roll = Mx / width / len(left_legs)
            for leg in left_legs:
                foot_forces[leg][2] -= dfz_roll
            for leg in right_legs:
                foot_forces[leg][2] += dfz_roll

        # Pitch moment → differential vertical force on front vs rear
        front_legs = [l for l in stance if l[0] == 'F']
        rear_legs  = [l for l in stance if l[0] == 'R']
        if front_legs and rear_legs:
            # Wheelbase ~0.376m
            wheelbase = 0.376
            dfz_pitch = My / wheelbase / len(front_legs)
            for leg in front_legs:
                foot_forces[leg][2] += dfz_pitch
            for leg in rear_legs:
                foot_forces[leg][2] -= dfz_pitch

        # Yaw moment → differential forward force on left vs right
        if left_legs and right_legs:
            # For yaw, left legs push forward (+Fx) while right legs push
            # backward (-Fx), creating a positive Mz (counterclockwise from top).
            dfx_yaw = Mz / width / len(left_legs)
            for leg in left_legs:
                foot_forces[leg][0] += dfx_yaw
            for leg in right_legs:
                foot_forces[leg][0] -= dfx_yaw

        # Ensure minimum vertical force (no pulling upward on foot)
        for leg in foot_forces:
            foot_forces[leg][2] = max(foot_forces[leg][2], -200.0)  # max downward = 200N
            foot_forces[leg][2] = min(foot_forces[leg][2], 0.0)     # no upward pull

        # ── Apply torques ──
        ramp = min(1.0, t / self._warm_up)

        for leg in ALL_LEGS:
            ctrl = self._ctrls[leg]
            ctrl.clear()
            state = self._scheduler.leg_state(leg, t)

            if state == "stance" and leg in foot_forces:
                # Stance: impedance around ground contact + feed-forward GRF
                target = ctrl.get_foot_pos().copy()
                target[2] = 0.0
                ff = foot_forces[leg]
                ctrl.apply_impedance(target, self._stance_Kp, self._stance_Kd,
                                     ff_force=ff)
            else:
                # Swing: track trajectory
                target_hip_xy = self._planner.get_target_hip_xy(leg, t)
                target_world_z = self._planner.get_target_world_z(leg, t)
                h_pos = ctrl.get_hip_pos()
                h_rot = ctrl.get_hip_rot()
                xy_world = h_pos[:2] + (h_rot[:2, :2] @ target_hip_xy)
                target = np.array([xy_world[0], xy_world[1], target_world_z])
                ctrl.apply_impedance(target, self._swing_Kp, self._swing_Kd)

    def step(self):
        self._sim.step()
