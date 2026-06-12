"""MIT-style locomotion controller for quadruped robots.

Architecture (based on MIT Cheetah / Mini Cheetah):
  1. Body PD: desired body wrench from height / roll / pitch / velocity errors
  2. Force distribution: allocate wrench to stance-foot ground reaction forces
  3. Joint torques: τ = Jᵀ · F  (Jacobian transpose)
  4. Swing legs: task-space PD tracking via Jᵀ

All torques are applied via MuJoCo's qfrc_applied, bypassing position actuators.
"""

import ctypes
import ctypes.util
import signal
from typing import Dict

import numpy as np

# ── SIGFPE guard for MuJoCo mj_step calls ────────────────────────────────────
# MuJoCo can trigger SIGFPE on certain configurations (intermittent on WSL2).

_libm = ctypes.CDLL(ctypes.util.find_library("m"))
_FE_ALL_EXCEPT = 0x01 | 0x04 | 0x08 | 0x10 | 0x20  # FE_INVALID|DIVBYZERO|OVERFLOW|UNDERFLOW|INEXACT


class _MuJoCoFPE(FloatingPointError):
    pass


def _sigfpe_handler(signum, frame):
    raise _MuJoCoFPE("MuJoCo triggered SIGFPE")


def _step_safe(sim) -> bool:
    """Call sim.step() — returns False if MuJoCo triggers SIGFPE."""
    prev_fe = _libm.fegetexcept()
    _libm.fedisableexcept(_FE_ALL_EXCEPT)
    prev_sig = signal.signal(signal.SIGFPE, _sigfpe_handler)
    try:
        sim.step()
        return True
    except _MuJoCoFPE:
        return False
    finally:
        signal.signal(signal.SIGFPE, prev_sig)
        _libm.feenableexcept(prev_fe)

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

    def jacobian_world(self, q: np.ndarray = None) -> np.ndarray:
        """3×3 Jacobian: q̇ → foot velocity in WORLD frame.

        Uses MuJoCo's built-in mj_jacSite for EXACT Jacobian, accounting for
        the thigh lateral offset [0, ±0.08, 0] that the analytical FK misses.
        """
        import mujoco
        site_id = self._sim._site_ids[self._ik._foot_site_name]
        nv = self._sim._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self._sim._model, self._sim._data, jacp, jacr, site_id)
        # Extract 3×3 submatrix for our 3 leg joints
        J = np.zeros((3, 3))
        for i, dof_addr in enumerate(self._dof_addrs):
            J[:, i] = jacp[:, dof_addr]
        return J

    def apply_force(self, F_world: np.ndarray):
        """Apply torque τ = Jᵀ · F to achieve foot force F in world frame."""
        J = self.jacobian_world()
        tau = J.T @ np.asarray(F_world)
        # Clamp torques to prevent numerical instability in MuJoCo
        tau = np.clip(tau, -35.0, 35.0)
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
        self.Kp_vy   = 50.0     # lateral velocity PD (N / (m/s))
        self.Kp_yaw  = 30.0     # yaw rate PD (Nm / (rad/s))
        self.target_vx = 0.3    # desired forward speed (m/s)
        self.target_vy = 0.0    # desired lateral speed (m/s)
        self.target_vyaw = 0.0  # desired yaw rate (rad/s)

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

    # ── Body PD wrench computation ─────────────────────────────────────────

    def _compute_body_pd_wrench(self, t: float) -> Dict[str, np.ndarray]:
        """Compute body PD wrench and distribute to stance feet.

        Returns:
            Dict mapping stance leg name → GRF [Fx, Fy, Fz] in world frame.
            Empty dict if no stance legs.
        """
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()

        target_height = 0.28

        Fz = -TOTAL_WEIGHT \
             + self.Kp_z * (pos[2] - target_height) \
             - self.Kd_z * vel[2]

        # ── Translational forces ──
        Fx = -self.Kp_vx * (self.target_vx - vel[0])
        Fy = -self.Kp_vy * (self.target_vy - vel[1])

        # ── Moments ──
        # Roll/pitch: disabled (impedance layer handles body-level angular
        # stability through foot placement — same as original code).
        # Yaw: PD tracking around target yaw rate
        Mx = 0.0
        My = 0.0
        Mz = -self.Kp_yaw * (self.target_vyaw - angvel[2])

        stance = self._scheduler.get_stance_legs(t)
        n = max(1, len(stance))

        foot_forces: Dict[str, np.ndarray] = {}

        stance_feet = []
        for leg in stance:
            p = self._ctrls[leg].get_foot_pos()
            stance_feet.append((leg, p))

        if len(stance_feet) == 0:
            return foot_forces

        # ── Even force distribution ──
        com = pos.copy()
        for leg, p_foot in stance_feet:
            f = np.zeros(3)
            f[0] = Fx / n
            f[1] = Fy / n
            f[2] = Fz / n
            foot_forces[leg] = f

        # ── Roll: differential Fz on left vs right ──
        left_legs  = [l for l in stance if l[1] == 'L']
        right_legs = [l for l in stance if l[1] == 'R']
        if left_legs and right_legs:
            width = 0.094
            dfz_roll = Mx / width / len(left_legs)
            for leg in left_legs:
                foot_forces[leg][2] -= dfz_roll
            for leg in right_legs:
                foot_forces[leg][2] += dfz_roll

        # ── Pitch: differential Fz on front vs rear ──
        front_legs = [l for l in stance if l[0] == 'F']
        rear_legs  = [l for l in stance if l[0] == 'R']
        if front_legs and rear_legs:
            wheelbase = 0.376
            dfz_pitch = My / wheelbase / len(front_legs)
            for leg in front_legs:
                foot_forces[leg][2] += dfz_pitch
            for leg in rear_legs:
                foot_forces[leg][2] -= dfz_pitch

        # ── Yaw: differential Fx on left vs right ──
        if left_legs and right_legs:
            width = 0.094
            dfx_yaw = Mz / width / len(left_legs)
            for leg in left_legs:
                foot_forces[leg][0] += dfx_yaw
            for leg in right_legs:
                foot_forces[leg][0] -= dfx_yaw

        # ── Final clipping ──
        for leg in foot_forces:
            foot_forces[leg][2] = max(foot_forces[leg][2], -200.0)
            foot_forces[leg][2] = min(foot_forces[leg][2], 0.0)

        return foot_forces

    # ── Apply impedance with feed-forward forces ───────────────────────────

    def _apply_leg_impedance(self, t: float,
                             foot_forces: Dict[str, np.ndarray]):
        """Apply per-leg impedance control with optional feed-forward force.

        Stance: τ = Jᵀ·(F_ff + Kp·Δx - Kd·v)  at z=0
        Swing:  τ = Jᵀ·(Kp·Δx - Kd·v)  tracking hip-frame trajectory
        """
        for leg in ALL_LEGS:
            ctrl = self._ctrls[leg]
            ctrl.clear()
            state = self._scheduler.leg_state(leg, t)

            if state == "stance" and leg in foot_forces:
                target = ctrl.get_foot_pos().copy()
                target[2] = 0.0
                ff = foot_forces[leg]
                ctrl.apply_impedance(target, self._stance_Kp, self._stance_Kd,
                                     ff_force=ff)
            else:
                target_hip_xy = self._planner.get_target_hip_xy(leg, t)
                target_world_z = self._planner.get_target_world_z(leg, t)
                h_pos = ctrl.get_hip_pos()
                h_rot = ctrl.get_hip_rot()
                xy_world = h_pos[:2] + (h_rot[:2, :2] @ target_hip_xy)
                target = np.array([xy_world[0], xy_world[1], target_world_z])
                ctrl.apply_impedance(target, self._swing_Kp, self._swing_Kd)

    # ── Main control loop ─────────────────────────────────────────────────

    def control(self, t: float):
        """Run one control step: Body PD → force distribution → impedance."""
        # Sync kinematic yaw target to foot trajectory planner
        self._planner.target_vyaw = self.target_vyaw
        foot_forces = self._compute_body_pd_wrench(t)
        self._apply_leg_impedance(t, foot_forces)

    def step(self):
        self._sim.step()


# ═══════════════════════════════════════════════════════════════════════════════
# MPC + MIT Impedance Controller
# ═══════════════════════════════════════════════════════════════════════════════

class MPCMITBodyController(MITBodyController):
    """MPC-based locomotion controller with MIT impedance control.

    Replaces the Body PD force distribution with SRB convex MPC:
      - MPC runs at ~31 Hz (every 16 sim steps at dt=0.002s)
      - MPC optimizes GRFs over a 0.3s horizon subject to SRB dynamics
        and friction cone constraints
      - Output GRFs are cached and fed as ff_force into the impedance layer
      - Swing legs continue with pure impedance trajectory tracking

    Falls back to Body PD if MPC QP fails to solve.

    Control law (per leg):
      τ = Jᵀ · (F_mpc + Kp · Δx - Kd · v)      [stance, MIT impedance]
      τ = Jᵀ · (Kp · Δx - Kd · v)               [swing, pure impedance]
    """

    # MPC runs every MPC_DECIMATION steps (16 steps × 0.002s = 0.032s ≈ 31 Hz)
    MPC_DECIMATION = 16

    def __init__(self, sim, gait_type, params=None, warm_up: float = 0.3):
        super().__init__(sim, gait_type, params, warm_up)

        # ── Softer stance gains for MPC (feed-forward handles most of the load) ──
        self._stance_Kp = np.array([80.0, 80.0, 200.0])
        self._stance_Kd = np.array([5.0, 5.0, 10.0])

        # ── Build SRB MPC solver ──
        from .mpc_controller import SrbMpcSolver

        # Hip offsets relative to trunk CoM (body frame)
        hip_offsets = {}
        for leg in ALL_LEGS:
            # Neutral hip position in body frame (computed at init)
            hip_offsets[leg] = self._neutral[leg].copy()

        self._mpc = SrbMpcSolver(
            mass=GO1_MASS,
            inertia_body=TRUNK_I,
            hip_offsets_com=hip_offsets,
            N=10, dt_mpc=0.03, mu=0.6,
            fz_min=30.0, fz_max=150.0,
        )

        # ── MPC state ──
        self._mpc_forces: Dict[str, np.ndarray] = {
            leg: np.zeros(3) for leg in ALL_LEGS
        }
        self._mpc_counter: int = 0
        self._mpc_fallback_count: int = 0
        self._mpc_total_count: int = 0
        self._mpc_solve_time: float = 0.0

        # ── MPC target velocities ──
        self.target_vy = 0.0
        self.target_vyaw = 0.0
        self.target_height = 0.28

    def control(self, t: float):
        """Run one control step with MPC + MIT impedance."""
        # Sync kinematic yaw target to foot trajectory planner
        self._planner.target_vyaw = self.target_vyaw
        step = self._mpc_counter

        # ── Run MPC at decimated rate (skip first few steps for stability) ──
        if step >= 8 and step % self.MPC_DECIMATION == 0:
            self._run_mpc(t)
        elif step < 8:
            # Startup: use Body PD for initial support
            pd = self._compute_body_pd_wrench(t)
            for leg in ALL_LEGS:
                if leg in pd:
                    self._mpc_forces[leg] = pd[leg]
                else:
                    self._mpc_forces[leg] = np.zeros(3)

        self._mpc_counter += 1

        # ── Apply impedance with MPC forces (or fallback) ──
        self._apply_leg_impedance(t, self._mpc_forces)

    def _run_mpc(self, t: float):
        """Solve MPC QP and cache GRF forces."""
        import time as _time

        # ── Get current body state ──
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()
        x0 = np.array([roll, pitch, yaw,
                       pos[0], pos[1], pos[2],
                       angvel[0], angvel[1], angvel[2],
                       vel[0], vel[1], vel[2]])

        # ── Get foot positions in world frame ──
        foot_positions = {}
        for leg in ALL_LEGS:
            foot_positions[leg] = self._ctrls[leg].get_foot_pos()

        # ── Contact schedule over horizon ──
        contact = self._mpc.compute_contact_schedule(t, self._scheduler)

        # ── Reference trajectory ──
        X_ref = self._mpc.build_reference(
            x0,
            target_vx=self.target_vx,
            target_vy=self.target_vy,
            target_vyaw=self.target_vyaw,
            target_height=self.target_height,
        )

        # ── Solve MPC ──
        t0 = _time.perf_counter()
        try:
            u_opt = self._mpc.solve(
                x0, X_ref, foot_positions,
                com_position=pos, yaw=yaw, contact=contact,
            )
        except Exception:
            u_opt = None
        elapsed = _time.perf_counter() - t0
        self._mpc_solve_time = elapsed
        self._mpc_total_count += 1

        if u_opt is not None:
            # MPC outputs ground reaction forces (GRF: ground→foot).
            # Impedance controller expects foot→ground forces (opposite sign).
            # Negate to match the impedance convention.
            mpc_forces = self._mpc.forces_to_dict(u_opt)
            max_ff = 200.0  # clip per-component feed-forward to safe range
            for leg in ALL_LEGS:
                f = -mpc_forces[leg]
                if not np.all(np.isfinite(f)):
                    f = np.zeros(3)
                f = np.clip(f, -max_ff, max_ff)
                # Ensure downward force (no upward pull on foot)
                f[2] = min(f[2], 0.0)
                f[2] = max(f[2], -max_ff)
                self._mpc_forces[leg] = f
        else:
            # Fallback to Body PD
            self._mpc_fallback_count += 1
            pd_forces = self._compute_body_pd_wrench(t)
            # Merge: MPC fails → use PD, keep swing legs zero
            for leg in ALL_LEGS:
                if leg in pd_forces:
                    self._mpc_forces[leg] = pd_forces[leg]
                else:
                    self._mpc_forces[leg] = np.zeros(3)

    @property
    def mpc_stats(self) -> dict:
        """Return MPC runtime statistics."""
        return {
            "solve_time_ms": self._mpc_solve_time * 1000,
            "fallback_count": self._mpc_fallback_count,
            "total_count": self._mpc_total_count,
            "fallback_rate": (self._mpc_fallback_count
                              / max(1, self._mpc_total_count)),
        }
