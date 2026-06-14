"""MIT-style locomotion controller for quadruped robots.

Architecture (based on MIT Cheetah / Mini Cheetah):
  1. Body PD: desired body wrench from height / roll / pitch / velocity errors
  2. Force distribution: allocate wrench to stance-foot ground reaction forces
  3. Joint torques: τ = Jᵀ · F  (Jacobian transpose)
  4. Swing legs: task-space PD tracking via Jᵀ

All torques are applied via MuJoCo's qfrc_applied, bypassing position actuators.
"""

from dataclasses import dataclass, field
from typing import List

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
# Velocity tracking metrics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VelocityMetrics:
    """Collects per-step velocity tracking data for offline analysis.

    Records body velocity, orientation, and tracking errors at each control
    step.  Provides summary statistics (steady-state mean, RMSE, etc.) for
    systematic controller tuning.
    """
    t: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)
    vz: List[float] = field(default_factory=list)
    wx: List[float] = field(default_factory=list)
    wy: List[float] = field(default_factory=list)
    wz: List[float] = field(default_factory=list)
    roll: List[float] = field(default_factory=list)
    pitch: List[float] = field(default_factory=list)
    height: List[float] = field(default_factory=list)
    vx_error: List[float] = field(default_factory=list)
    vy_error: List[float] = field(default_factory=list)
    wz_error: List[float] = field(default_factory=list)

    def record(self, t: float, vel, angvel, roll, pitch, height,
               target_vx: float, target_vy: float, target_vyaw: float):
        self.t.append(t)
        self.vx.append(float(vel[0]))
        self.vy.append(float(vel[1]))
        self.vz.append(float(vel[2]))
        self.wx.append(float(angvel[0]))
        self.wy.append(float(angvel[1]))
        self.wz.append(float(angvel[2]))
        self.roll.append(float(roll))
        self.pitch.append(float(pitch))
        self.height.append(float(height))
        self.vx_error.append(float(target_vx - vel[0]))
        self.vy_error.append(float(target_vy - vel[1]))
        self.wz_error.append(float(target_vyaw - angvel[2]))

    def summary(self) -> dict:
        """Return key tracking metrics.

        Uses the last 30% of data for steady-state analysis, which excludes
        startup transients (settle, warm-up ramp).
        """
        if len(self.t) < 50:
            return {"error": "insufficient data (< 50 samples)"}

        # Steady-state: last 30% of trajectory
        n_ss = max(int(len(self.t) * 0.3), 50)
        vx_arr = np.array(self.vx[-n_ss:])
        vy_arr = np.array(self.vy[-n_ss:])
        wz_arr = np.array(self.wz[-n_ss:])
        vx_err = np.array(self.vx_error[-n_ss:])
        vy_err = np.array(self.vy_error[-n_ss:])
        wz_err = np.array(self.wz_error[-n_ss:])
        roll_arr = np.array(self.roll[-n_ss:])
        pitch_arr = np.array(self.pitch[-n_ss:])
        h_arr = np.array(self.height[-n_ss:])

        # Tracking percentages
        mean_vx = float(np.mean(vx_arr))
        mean_vy = float(np.mean(vy_arr))
        mean_wz = float(np.mean(wz_arr))

        # Use the latest target values for percentage (targets stored in errors)
        target_vx_used = mean_vx + float(np.mean(vx_err))
        target_vy_used = mean_vy + float(np.mean(vy_err))
        target_wz_used = mean_wz + float(np.mean(wz_err))

        def track_pct(mean_val, target_val):
            if abs(target_val) < 0.01:
                return 100.0  # tracking zero is trivially perfect
            return max(0.0, 100.0 * (1.0 - abs(target_val - mean_val) / abs(target_val)))

        return {
            "n_samples": len(self.t),
            "n_steady": n_ss,
            "vx_mean": mean_vx,
            "vy_mean": mean_vy,
            "wz_mean": mean_wz,
            "vx_target": target_vx_used,
            "vy_target": target_vy_used,
            "wz_target": target_wz_used,
            "vx_track_pct": track_pct(mean_vx, target_vx_used),
            "vy_track_pct": track_pct(mean_vy, target_vy_used),
            "wz_track_pct": track_pct(mean_wz, target_wz_used),
            "vx_rmse": float(np.sqrt(np.mean(vx_err ** 2))),
            "vy_rmse": float(np.sqrt(np.mean(vy_err ** 2))),
            "wz_rmse": float(np.sqrt(np.mean(wz_err ** 2))),
            "height_mean": float(np.mean(h_arr)),
            "height_std": float(np.std(h_arr)),
            "roll_rms_deg": float(np.sqrt(np.mean(roll_arr ** 2)) * 180 / np.pi),
            "pitch_rms_deg": float(np.sqrt(np.mean(pitch_arr ** 2)) * 180 / np.pi),
            "vx_final": float(vx_arr[-1]) if len(vx_arr) > 0 else 0.0,
            "vy_final": float(vy_arr[-1]) if len(vy_arr) > 0 else 0.0,
        }

    def print_summary(self):
        """Pretty-print tracking metrics."""
        s = self.summary()
        if "error" in s:
            print(f"  [Metrics] {s['error']}")
            return
        print(f"\n{'─'*60}")
        print(f"  Velocity Tracking Summary (steady-state, last {s['n_steady']} steps)")
        print(f"  {'─'*50}")
        print(f"  vx:  {s['vx_mean']:+.3f} / {s['vx_target']:+.3f} m/s  "
              f"→ {s['vx_track_pct']:.1f}%  (RMSE {s['vx_rmse']:.3f})")
        print(f"  vy:  {s['vy_mean']:+.3f} / {s['vy_target']:+.3f} m/s  "
              f"→ {s['vy_track_pct']:.1f}%  (RMSE {s['vy_rmse']:.3f})")
        print(f"  wz:  {s['wz_mean']:+.3f} / {s['wz_target']:+.3f} rad/s "
              f"→ {s['wz_track_pct']:.1f}%  (RMSE {s['wz_rmse']:.3f})")
        print(f"  height: {s['height_mean']:.3f} m  (σ={s['height_std']:.3f})")
        print(f"  roll:  {s['roll_rms_deg']:.1f}° RMS  "
              f"pitch: {s['pitch_rms_deg']:.1f}° RMS")
        print(f"{'─'*60}\n")


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
        self.Kp_roll = 30.0     # roll (Nm/rad)
        self.Kd_roll = 10.0     # roll damping
        self.Kp_pitch = 30.0    # pitch
        self.Kd_pitch = 10.0
        self.Kp_vx   = 500.0    # forward velocity (N / (m/s))
        self.Kp_vy   = 200.0    # lateral velocity PD (N / (m/s))
        self.Kp_yaw  = 30.0     # yaw rate PD (Nm / (rad/s))
        self.Kp_px   = 20.0     # CoM position anchor stiffness (N/m)
        self.Kp_py   = 20.0     # CoM position anchor stiffness (N/m)
        self.target_vx = 0.3    # desired forward speed (m/s)
        self.target_vy = 0.0    # desired lateral speed (m/s)
        self.target_vyaw = 0.0  # desired yaw rate (rad/s)

        # Stance / swing gains
        self._stance_Kp = np.array([150.0, 150.0, 500.0])
        self._stance_Kd = np.array([10.0,  10.0,  20.0])
        self._swing_Kp  = np.array([400.0, 400.0, 400.0])
        self._swing_Kd  = np.array([15.0,  15.0,  15.0])

        # ── Body-frame stance offset (for CoM-centric rotation) ──
        # On swing→stance transition, snapshot the body-frame foot offset.
        # During stance the foot orbits around the CURRENT CoM position,
        # preventing the foot from locking to a fixed world point (which
        # forces the body to translate during rotation).
        self._stance_offset_body: Dict[str, np.ndarray] = {
            leg: np.zeros(3) for leg in ALL_LEGS
        }
        self._com_anchor: np.ndarray = None  # set after settle
        self._last_gait_state: Dict[str, str] = {
            leg: "stance" for leg in ALL_LEGS
        }

        # ── Velocity tracking metrics ──
        self._metrics = VelocityMetrics()
        self._metrics_enabled = True

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

    @staticmethod
    def _quat_to_rotmat(quat):
        """Convert quaternion [w, x, y, z] to 3×3 rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])

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

        # ── CoM position anchor (resist drift during pure rotation) ──
        # Only active when not commanded to translate or yaw.
        if abs(self.target_vx) < 0.01 and abs(self.target_vy) < 0.01 \
                and abs(self.target_vyaw) < 0.01:
            if self._com_anchor is None:
                self._com_anchor = pos[:2].copy()
            Fx += -self.Kp_px * (pos[0] - self._com_anchor[0])
            Fy += -self.Kp_py * (pos[1] - self._com_anchor[1])

        # ── Moments ──
        # Roll/pitch: active PD to keep body level.
        # -Kp * angle - Kd * angvel damps body oscillations.
        # Yaw: PD tracking around target yaw rate.
        Mx = -self.Kp_roll * roll - self.Kd_roll * angvel[0]
        My = -self.Kp_pitch * pitch - self.Kd_pitch * angvel[1]
        Mz = self.Kp_yaw * (self.target_vyaw - angvel[2])

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

        Stance: foot target orbits around current CoM (pos + R_body·offset),
                captured on swing→stance transition.  This keeps the foot
                rotating with the body instead of locking to a fixed world
                point (which forces the CoM to translate).
        Swing:  τ = Jᵀ·(Kp·Δx - Kd·v)  tracking hip-frame trajectory
        """
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()
        R_body = self._quat_to_rotmat(quat)

        for leg in ALL_LEGS:
            ctrl = self._ctrls[leg]
            ctrl.clear()
            state = self._scheduler.leg_state(leg, t)
            prev = self._last_gait_state.get(leg, "stance")

            if state == "stance" and leg in foot_forces:
                # ── Capture body-frame offset on swing→stance transition ──
                if prev == "swing" or np.allclose(self._stance_offset_body[leg], 0):
                    foot_world = ctrl.get_foot_pos()
                    self._stance_offset_body[leg] = R_body.T @ (foot_world - pos)

                # ── Target = current CoM + current rotation @ body offset ──
                target = pos + R_body @ self._stance_offset_body[leg]
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

            self._last_gait_state[leg] = state

    # ── Main control loop ─────────────────────────────────────────────────

    def control(self, t: float):
        """Run one control step: Body PD → force distribution → impedance."""
        # Sync kinematic yaw target to foot trajectory planner
        self._planner.target_vyaw = self.target_vyaw
        foot_forces = self._compute_body_pd_wrench(t)
        self._apply_leg_impedance(t, foot_forces)

        # ── Record velocity tracking metrics ──
        if self._metrics_enabled:
            pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()
            self._metrics.record(t, vel, angvel, roll, pitch, pos[2],
                                self.target_vx, self.target_vy, self.target_vyaw)

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

        # ── Record velocity tracking metrics ──
        if self._metrics_enabled:
            pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()
            self._metrics.record(t, vel, angvel, roll, pitch, pos[2],
                                self.target_vx, self.target_vy, self.target_vyaw)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Quintic + Friction Force Controller
# ═══════════════════════════════════════════════════════════════════════════════

class QuinticFrictionController(MITBodyController):
    """Locomotion controller using quintic swing trajectories and
    static-friction-constrained stance force distribution.

    Architecture (three tiers):
      Tier 1 (~10 Hz):  Gait parameter adaptation — online gradient descent on
                        (T_cycle, step_length) to minimise friction cost while
                        tracking target velocity.
      Tier 2 (on demand): Quintic (5th-order) polynomial swing trajectories
                        with C² continuity at touchdown and liftoff.
      Tier 3 (~500 Hz): Static-friction-constrained force distribution —
                        minimum-norm solution in the 1-D null space of the
                        underdetermined 3×4 system (2 stance legs × 2 axes).

    Inherits MITBodyController for:
      - LegTorqueController setup and Jacobian-transpose torque application
      - Settle behaviour (position-actuator warm-up)
      - Body state queries and quaternion→RPY conversion
    """

    # Tier-1 adaptation interval (in sim steps at dt=0.002s → ~10 Hz)
    ADAPT_INTERVAL = 50  # every 0.1 s

    def __init__(self, sim, gait_type,
                 params: GaitParams = None, warm_up: float = 0.3,
                 mu_max: float = 0.6, adapt_params: bool = False):
        super().__init__(sim, gait_type, params, warm_up)

        # ── Replace planner with quintic variant ──
        from .gait import QuinticFootTrajectoryPlanner
        self._planner = QuinticFootTrajectoryPlanner(
            self._scheduler, self._neutral, warm_up=warm_up,
        )

        # ── Friction force distributor ──
        from .friction_force import FrictionForceDistributor
        self._friction = FrictionForceDistributor(mu_max=mu_max)

        # ── Kinematic velocity control gains (Raibert-style foot placement) ──
        # Adjust foot landing to create favorable geometry for the force PD.
        self._planner.K_kin_vx = 1.0   # step_length delta per m/s vx error
        self._planner.K_kin_vy = 0.4   # lateral offset per m/s vy error
        self._planner.K_kin_wz = 2.0   # yaw look-ahead time (s): θ = K * wz_err

        # ── Force-based velocity gains (provides the actual push) ──
        # Kinematic placement gives geometric advantage;
        # these gains provide the feed-forward force through the favourable geometry.
        self.Kp_vx = 500.0   # forward push
        self.Kp_vy = 200.0   # lateral push
        self.Kp_yaw = 15.0   # yaw torque

        # ── Stance impedance: soft enough to let ff_force push foot back ──
        self._stance_Kp = np.array([100.0, 100.0, 500.0])
        self._stance_Kd = np.array([5.0,   5.0,   20.0])

        # ── CoM anchor: moderate stiffness, disabled during active yaw ──
        self.Kp_px = 100.0
        self.Kp_py = 100.0

        # ── Yaw angle tracking (P + I to eliminate steady-state error) ──
        self._initial_yaw: float = None  # set on first control step after settle
        self._prev_yaw: float = 0.0     # for world-frame wz computation
        self._prev_t: float = 0.0
        self._world_wz: float = 0.0     # true world-frame yaw rate (from quat diff)
        self._K_yaw_angle_kin = 4.0     # kinematic yaw angle P-gain (rad/rad)
        self._K_yaw_angle_force = 30.0  # force yaw angle P-gain (Nm/rad)
        self._K_yaw_angle_int = 40.0    # force yaw angle I-gain (Nm/rad·s)
        self._yaw_error_integral: float = 0.0

        # ── Tier-1 adaptation state ──
        self._adapt_params = adapt_params
        self._adapt_counter: int = 0
        self._mu_history: list[float] = []       # recent μ_utilized values
        self._vx_error_history: list[float] = []  # recent vx errors

        # Adaptation hyper-parameters
        self._adapt_lr_L = 0.005     # step-length learning rate (m)
        self._adapt_lr_T = 0.002     # cycle-period learning rate (s)
        self._adapt_w_v = 10.0       # velocity error weight in cost
        self._adapt_w_mu = 1.0       # friction utilisation weight
        self._adapt_w_T = 0.5        # cycle-period regularisation

        # Parameter bounds
        self._L_min, self._L_max = 0.02, 0.35   # step length (m)
        self._T_min, self._T_max = 0.10, 0.80   # cycle period (s)

    @property
    def mu_utilized(self) -> float:
        """Current friction utilisation ratio (0 = no friction used, 1 = at limit)."""
        return self._friction.mu_utilized

    @property
    def friction_feasible(self) -> bool:
        """Whether the last force distribution was fully feasible."""
        return self._friction.feasible

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ── Tier 3: Friction-constrained force computation ──────────────────────

    def _compute_friction_forces(self, t: float) -> Dict[str, np.ndarray]:
        """Compute body PD wrench and distribute to stance feet via friction
        constraints (replaces the heuristic even-split distribution).

        Returns:
            Dict mapping stance leg name → GRF [Fx, Fy, Fz] in world frame.
        """
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()

        target_height = 0.28

        # ── Body PD wrench (same as MITBodyController) ──
        Fz = -TOTAL_WEIGHT \
             + self.Kp_z * (pos[2] - target_height) \
             - self.Kd_z * vel[2]

        # Fx, Fy: -Kp * error → when vel < target, F positive (forward push)
        Fx = -self.Kp_vx * (self.target_vx - vel[0])
        Fy = -self.Kp_vy * (self.target_vy - vel[1])
        # Roll/pitch: active PD to keep body level
        Mx = -self.Kp_roll * roll - self.Kd_roll * angvel[0]
        My = -self.Kp_pitch * pitch - self.Kd_pitch * angvel[1]
        # Mz: rate PD + angle P + integral (base gain always active, scaled up for lateral)
        yaw_err_angle = getattr(self, '_yaw_angle_error', 0.0)
        yaw_err_int = getattr(self, '_yaw_error_integral', 0.0)
        lat_ratio = abs(self.target_vy) / max(abs(self.target_vx) + abs(self.target_vy), 0.01)
        # Base yaw P (always active) + lateral boost
        yaw_P_eff = 5.0 + 20.0 * lat_ratio   # 5→25 Nm/rad
        yaw_I_eff = 2.0 + 38.0 * lat_ratio   # 2→40 Nm/rad·s
        Mz = (self.Kp_yaw * (self.target_vyaw - angvel[2])
              + yaw_P_eff * yaw_err_angle
              + yaw_I_eff * yaw_err_int)

        # ── CoM position anchor (resist drift during pure rotation) ──
        if abs(self.target_vx) < 0.01 and abs(self.target_vy) < 0.01:
            if self._com_anchor is None:
                self._com_anchor = pos[:2].copy()
            Fx += -self.Kp_px * (pos[0] - self._com_anchor[0])
            Fy += -self.Kp_py * (pos[1] - self._com_anchor[1])

        stance = self._scheduler.get_stance_legs(t)
        n = max(1, len(stance))

        if n == 0:
            return {}

        # ── Yaw-prioritized wrench: reduce lateral force when yaw error is large ──
        yaw_err = abs(getattr(self, '_yaw_angle_error', 0.0))
        if yaw_err > 0.15:  # only trigger above ~9 degrees
            yaw_priority = min(1.0, (yaw_err - 0.15) / 0.3)  # 0→1 over 9°→26°
            Fx *= (1.0 - 0.7 * yaw_priority)
            Fy *= (1.0 - 0.7 * yaw_priority)

        # ── Vertical force: even distribution (gravity compensation) ──
        fz_per_leg = Fz / n
        foot_fz = {leg: fz_per_leg for leg in stance}

        # ── Roll/Pitch: differential Fz for active body leveling ──
        left_legs  = [l for l in stance if l[1] == 'L']
        right_legs = [l for l in stance if l[1] == 'R']
        if left_legs and right_legs:
            width = 0.094
            dfz_roll = Mx / width / len(left_legs)
            for leg in left_legs:
                foot_fz[leg] -= dfz_roll
            for leg in right_legs:
                foot_fz[leg] += dfz_roll
        front_legs = [l for l in stance if l[0] == 'F']
        rear_legs  = [l for l in stance if l[0] == 'R']
        if front_legs and rear_legs:
            wheelbase = 0.376
            dfz_pitch = My / wheelbase / len(front_legs)
            for leg in front_legs:
                foot_fz[leg] += dfz_pitch
            for leg in rear_legs:
                foot_fz[leg] -= dfz_pitch

        # ── Horizontal forces: friction-constrained distribution ──
        foot_positions = {}
        for leg in stance:
            foot_positions[leg] = self._ctrls[leg].get_foot_pos()

        # Use estimated whole-body COM (trunk inertial offset + leg mass averaging).
        # Trunk body origin is NOT the COM. True COM is ~1cm forward, ~15cm down.
        R_body = self._quat_to_rotmat(quat)
        com_offset_body = np.array([0.01, 0.0, -0.15])  # rough whole-body estimate
        com_actual = pos + R_body @ com_offset_body

        horizontal_forces = self._friction.distribute(
            desired_wrench=np.array([Fx, Fy, Mz]),
            foot_positions=foot_positions,
            foot_fz=foot_fz,
            stance_legs=stance,
            com_position=com_actual,
        )

        # ── Assemble full 3D forces ──
        foot_forces: Dict[str, np.ndarray] = {}
        for leg in stance:
            fh = horizontal_forces.get(leg, np.zeros(2))
            fz_val = fz_per_leg
            foot_forces[leg] = np.array([fh[0], fh[1], fz_val])

        # ── Clip vertical force to safe range ──
        for leg in foot_forces:
            foot_forces[leg][2] = np.clip(foot_forces[leg][2], -200.0, 0.0)

        return foot_forces

    # ── Main control loop ─────────────────────────────────────────────────

    def control(self, t: float):
        """Run one control step: kinematic foot placement → friction forces → impedance.

        1. Measure velocity errors
        2. Adjust foot landing positions (Raibert-style kinematic control)
        3. Compute friction-constrained support forces (reduced force PD)
        4. Apply impedance with quintic swing trajectories
        """
        # ── Step 1: Measure velocity errors + yaw angle ──
        pos, _, vel, angvel, _, _, yaw = self._body_state()

        # Track yaw angle relative to initial heading
        if self._initial_yaw is None:
            self._initial_yaw = yaw
            self._prev_yaw = yaw
            self._prev_t = t
        desired_yaw = self._initial_yaw + self.target_vyaw * t
        yaw_angle_error = self._wrap_pi(desired_yaw - yaw)

        # Compute TRUE world-frame yaw rate from quaternion (qvel[5] is NOT world wz!)
        dt_yaw = t - self._prev_t
        if dt_yaw > 1e-9:
            self._world_wz = self._wrap_pi(yaw - self._prev_yaw) / dt_yaw
        self._prev_yaw = yaw
        self._prev_t = t

        # Integrate yaw error — base rate always, faster for lateral motion
        lat_ratio = abs(self.target_vy) / max(abs(self.target_vx) + abs(self.target_vy), 0.01)
        int_rate = 0.2 + 0.8 * lat_ratio  # 0.2→1.0 scaling
        self._yaw_error_integral += yaw_angle_error * 0.002 * int_rate
        self._yaw_error_integral = float(np.clip(
            self._yaw_error_integral, -2.0, 2.0))  # anti-windup

        dvx = self.target_vx - vel[0]
        dvy = self.target_vy - vel[1]
        dwz = self.target_vyaw - angvel[2]  # qvel rate damping (not _world_wz)

        # ── Step 2: Kinematic foot placement (PRIMARY velocity control) ──
        step_delta = np.clip(self._planner.K_kin_vx * dvx, -0.15, 0.15)
        lat_offset = np.clip(self._planner.K_kin_vy * dvy, -0.05, 0.05)
        # Yaw: rate + angle P + integral (base always active, scaled up for lateral)
        lat_ratio = abs(self.target_vy) / max(abs(self.target_vx) + abs(self.target_vy), 0.01)
        k_angle = 0.5 + 3.5 * lat_ratio   # 0.5→4.0 rad/rad
        k_int = 1.0 + 4.0 * lat_ratio     # 1.0→5.0 rad/rad·s
        yaw_offset = np.clip(
            self._planner.K_kin_wz * dwz
            + k_angle * yaw_angle_error
            + k_int * self._yaw_error_integral,
            -0.8, 0.8)
        self._planner.set_kinematic_adjustments(step_delta, lat_offset, yaw_offset)

        # Sync kinematic yaw target (disabled, using yaw_offset instead)
        self._planner.target_vyaw = 0.0

        # Store yaw angle error for use in force computation
        self._yaw_angle_error = yaw_angle_error

        # ── Step 3: Friction-constrained forces (SUPPORT + complementary PD) ──
        foot_forces = self._compute_friction_forces(t)

        # ── Step 4: Impedance control with quintic trajectories ──
        self._apply_leg_impedance(t, foot_forces)

        # Direct yaw torque: only for lateral-dominant motion where leg chain
        # has poor yaw authority. Skip for forward walking.
        if abs(self.target_vy) > 0.05 and abs(self.target_vx) < 0.05:
            yaw_err2 = getattr(self, '_yaw_angle_error', 0.0)
            yaw_int2 = getattr(self, '_yaw_error_integral', 0.0)
            Mz_direct = 2.0 * yaw_err2 + 3.0 * yaw_int2
            trunk_id = self._sim._body_ids.get("trunk")
            if trunk_id is not None:
                jnt_id = self._sim._model.body_jntadr[trunk_id]
                dof_adr = self._sim._model.jnt_dofadr[jnt_id]
                self._sim._data.qfrc_applied[dof_adr + 5] = float(np.clip(
                    Mz_direct, -5.0, 5.0))

        # Record stats for Tier-1 adaptation
        self.record_stats()

        # ── Record velocity tracking metrics ──
        if self._metrics_enabled:
            pos, _, vel, angvel, roll, pitch, _ = self._body_state()
            self._metrics.record(t, vel, angvel, roll, pitch, pos[2],
                                self.target_vx, self.target_vy, self.target_vyaw)

        # Tier 1: periodic gait-parameter adaptation
        if self._adapt_params:
            self._adapt_counter += 1
            if self._adapt_counter >= self.ADAPT_INTERVAL:
                self._adapt_counter = 0
                self._adapt_gait_params()


    def step(self):
        self._sim.step()

    # ── Tier 1: Gait parameter adaptation ──────────────────────────────────

    def _adapt_gait_params(self):
        """Online adaptation of step_length to balance velocity tracking
        and friction utilisation.

        Simple rule-based approach:
          - If μ_util < 0.85 and v_actual < v_target: slightly increase
            step_length (gives feet more push distance per stance).
          - If μ_util > 0.95 (near friction limit): slightly decrease
            step_length to reduce force demand.
          - T_cycle is kept at the user-specified value.
        """
        if len(self._mu_history) < 5 or len(self._vx_error_history) < 5:
            return

        mu_avg = np.mean(self._mu_history[-5:])
        vx_err_avg = np.mean(self._vx_error_history[-5:])
        _, _, vel, _, _, _, _ = self._body_state()

        T = self._scheduler.T_cycle
        L = self._scheduler.params.step_length

        # Adaptation step sizes
        dL_up = 0.005     # increase when under speed with margin
        dL_down = 0.003   # decrease when near friction limit

        if mu_avg < 0.85 and vel[0] < self.target_vx * 0.9:
            # Plenty of friction margin but not tracking velocity → increase L
            L_new = L + dL_up
        elif mu_avg > 0.95:
            # Near friction limit → back off slightly
            L_new = L - dL_down
        else:
            return  # no change needed

        L_new = np.clip(L_new, self._L_min, self._L_max)

        if abs(L_new - L) > 0.0005:
            self._scheduler._params.step_length = float(L_new)
            self._planner._swing_duration = (
                (1.0 - self._scheduler.duty_factor) * T
            )

    # ── Statistics ─────────────────────────────────────────────────────────

    def record_stats(self):
        """Record current friction utilisation and velocity error for Tier-1."""
        _, _, vel, _, _, _, _ = self._body_state()
        self._mu_history.append(self._friction.mu_utilized)
        self._vx_error_history.append(self.target_vx - vel[0])
        # Keep bounded
        if len(self._mu_history) > 200:
            self._mu_history = self._mu_history[-100:]
            self._vx_error_history = self._vx_error_history[-100:]

    @property
    def friction_stats(self) -> dict:
        """Return friction-controller statistics."""
        return {
            "mu_utilized": self._friction.mu_utilized,
            "feasible": self._friction.feasible,
            "nullspace_alpha": self._friction.last_alpha,
            "T_cycle": self._scheduler.T_cycle,
            "step_length": self._scheduler.params.step_length,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum-based Force Controller
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumController(QuinticFrictionController):
    """Locomotion controller using 6×6 Newton-Euler force distribution.

    Instead of Body PD + heuristic force split (parent), solves the full
    Newton-Euler equation for N stance legs:

        [F_net]   =   [ I_3      I_3    ... ] [F1]
        [M_net]       [[r1]_×  [r2]_×  ... ] [F2]
                                              [..]

    For trot/pace (2 stance legs): direct 6×6 solve.
    For walk (3+ legs): damped least squares.
    Swing legs: impedance-only quintic trajectory tracking (inherited).

    Inherits from QuinticFrictionController:
      - QuinticFootTrajectoryPlanner (C² swing trajectories)
      - LegTorqueController + Jacobian-transpose torque application
      - Stance foot offset capture with body-rotation following
      - Tier-1 gait parameter adaptation
      - Settle behaviour (position-actuator warm-up)
    """

    def __init__(self, sim, gait_type,
                 params: GaitParams = None, warm_up: float = 0.3,
                 mu_max: float = 0.6, adapt_params: bool = False):
        super().__init__(sim, gait_type, params, warm_up, mu_max, adapt_params)

        # ── Momentum-specific gains (velocity/position error → acceleration) ──
        self._K_m_vx = 25.0     # forward accel gain (m/s² per m/s error)
        self._K_m_vy = 12.0     # lateral accel gain
        self._K_m_pz = 200.0    # height PD proportional (1/s²)
        self._K_m_dz = 40.0     # height PD derivative (1/s)

        self._K_m_roll = 80.0    # roll PD proportional (1/s²)
        self._K_m_droll = 20.0   # roll PD derivative (1/s)
        self._K_m_pitch = 100.0  # pitch PD proportional (1/s²)
        self._K_m_dpitch = 25.0  # pitch PD derivative (1/s)
        self._K_m_yaw = 8.0      # yaw rate PD (1/s)

        # ── Friction enforcement after 6×6 solve ──
        self.momentum_mu = mu_max

        # ── Least-squares damping ──
        self._lsq_damping = 0.01

        # ── Force limits ──
        self._max_leg_force = 250.0
        self._max_fz = 250.0

        # ── Stats tracking ──
        self._momentum_mu_max: float = 0.0
        self._momentum_feasible: bool = True
        self._momentum_cond = None  # type: float | None
        self._momentum_n_stance: int = 0

    # ── Newton-Euler tools ────────────────────────────────────────────────

    @staticmethod
    def _skew(r: np.ndarray) -> np.ndarray:
        """Cross-product matrix: [r]_× v = r × v."""
        return np.array([
            [0.0,    -r[2],   r[1]],
            [r[2],    0.0,   -r[0]],
            [-r[1],   r[0],   0.0],
        ])

    def _compute_desired_wrench(self) -> np.ndarray:
        """Compute 6-DOF desired wrench in ground→foot convention.

        Fz > 0 means upward force on feet (counteracts gravity).
        """
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()
        target_height = 0.28

        # ── Linear: m·a_des + [0, 0, mg] ──
        ax = self._K_m_vx * (self.target_vx - vel[0])
        ay = self._K_m_vy * (self.target_vy - vel[1])
        az = self._K_m_pz * (target_height - pos[2]) - self._K_m_dz * vel[2]

        Fx = GO1_MASS * ax
        Fy = GO1_MASS * ay
        Fz = TOTAL_WEIGHT + GO1_MASS * az

        # CoM anchor for pure-rotation mode
        if abs(self.target_vx) < 0.01 and abs(self.target_vy) < 0.01 \
                and abs(self.target_vyaw) < 0.01:
            if self._com_anchor is None:
                self._com_anchor = pos[:2].copy()
            Fx += -self.Kp_px * (pos[0] - self._com_anchor[0])
            Fy += -self.Kp_py * (pos[1] - self._com_anchor[1])

        # ── Angular: I·α_des ──
        alphax = self._K_m_roll * (-roll) - self._K_m_droll * angvel[0]
        alphay = self._K_m_pitch * (-pitch) - self._K_m_dpitch * angvel[1]
        # Yaw: rate PD + angle P+I (world-frame wz from quaternion)
        yaw_err_angle = getattr(self, '_yaw_angle_error', 0.0)
        yaw_err_int = getattr(self, '_yaw_error_integral', 0.0)
        world_wz = getattr(self, '_world_wz', angvel[2])
        alphaz = (self._K_m_yaw * (self.target_vyaw - world_wz)
                  + 3.0 * yaw_err_angle + 2.0 * yaw_err_int)

        Mx = TRUNK_I[0] * alphax
        My = TRUNK_I[1] * alphay
        Mz = TRUNK_I[2] * alphaz

        return np.array([Fx, Fy, Fz, Mx, My, Mz])

    # ── Solver variants ───────────────────────────────────────────────────

    def _solve_6x6(self, W_des: np.ndarray,
                   r: dict, legs: list) -> np.ndarray:
        """Direct solve A·f = W_des for exactly 2 stance legs.

        Falls back to damped LS if the matrix is poorly conditioned.

        Returns:
            f_vec = [F1x, F1y, F1z, F2x, F2y, F2z] in ground→foot convention.
        """
        r1, r2 = r[legs[0]], r[legs[1]]

        A = np.zeros((6, 6))
        A[0:3, 0:3] = np.eye(3)
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = self._skew(r1)
        A[3:6, 3:6] = self._skew(r2)

        cond = float(np.linalg.cond(A))
        self._momentum_cond = cond

        if cond > 1e6 or not np.all(np.isfinite(A)):
            damping = 1e-3 * np.eye(6)
            return np.linalg.solve(A.T @ A + damping, A.T @ W_des)

        return np.linalg.solve(A, W_des)

    def _solve_lsq(self, W_des: np.ndarray,
                   r: dict, legs: list) -> np.ndarray:
        """Damped least squares for 1 or 3+ stance legs.

        Builds A (6×3N), solves (AᵀA + λI)·f = Aᵀ·W_des.
        """
        n = len(legs)
        A = np.zeros((6, 3 * n))
        for i, leg in enumerate(legs):
            col = 3 * i
            A[0:3, col:col + 3] = np.eye(3)
            A[3:6, col:col + 3] = self._skew(r[leg])

        lam = self._lsq_damping * np.eye(3 * n)
        f_vec = np.linalg.solve(A.T @ A + lam, A.T @ W_des)

        for i in range(n):
            idx = 3 * i
            f_i = f_vec[idx:idx + 3]
            f_norm = np.linalg.norm(f_i)
            if f_norm > self._max_leg_force:
                f_vec[idx:idx + 3] *= self._max_leg_force / f_norm

        return f_vec

    # ── Friction enforcement ──────────────────────────────────────────────

    @staticmethod
    def _max_mu_ratio(f_vec: np.ndarray, n: int, mu: float) -> float:
        """Max friction utilisation across all legs."""
        ratios = []
        for i in range(n):
            idx = 3 * i
            f_h = np.linalg.norm(f_vec[idx:idx + 2])
            f_z = max(abs(f_vec[idx + 2]), 1.0)
            ratios.append(f_h / (mu * f_z))
        return max(ratios) if ratios else 0.0

    def _enforce_friction(self, f_vec: np.ndarray, W_des: np.ndarray,
                          r: dict, legs: list):
        """Check friction; scale [Fx,Fy,Mx,My,Mz] if any leg exceeds μ·|Fz|.

        Returns:
            (f_vec_final, feasible: bool, mu_max_used: float)
        """
        n = len(legs)
        mu = self.momentum_mu
        mu_max = self._max_mu_ratio(f_vec, n, mu)

        if mu_max <= 1.0 + 1e-6:
            return f_vec.copy(), True, mu_max

        # Binary search λ ∈ [0, 1] on horizontal + rotational wrench
        lo, hi = 0.0, 1.0
        best = f_vec.copy()

        for _ in range(20):
            mid = (lo + hi) / 2.0
            W_scaled = W_des.copy()
            W_scaled[0] *= mid   # Fx
            W_scaled[1] *= mid   # Fy
            # W_scaled[2] kept (Fz — need gravity support)
            W_scaled[3] *= mid   # Mx
            W_scaled[4] *= mid   # My
            W_scaled[5] *= mid   # Mz

            if n == 2:
                f_try = self._solve_6x6(W_scaled, r, legs)
            else:
                f_try = self._solve_lsq(W_scaled, r, legs)

            if self._max_mu_ratio(f_try, n, mu) <= 1.0 + 1e-6:
                best = f_try.copy()
                lo = mid
            else:
                hi = mid

        return best, (lo > 0.0), self._max_mu_ratio(best, n, mu)

    # ── Main force computation ────────────────────────────────────────────

    def _compute_momentum_forces(self, t: float) -> Dict[str, np.ndarray]:
        """Full 6-DOF momentum-based stance force distribution.

        1. Desired CoM wrench from body PD (ground→foot)
        2. Assemble Newton-Euler matrix A (6×3N)
        3. Solve for foot forces
        4. Enforce friction constraints
        5. Negate → foot→ground convention for impedance layer
        """
        pos, quat, vel, angvel, roll, pitch, yaw = self._body_state()
        stance = self._scheduler.get_stance_legs(t)
        n = len(stance)
        self._momentum_n_stance = n

        if n == 0:
            return {}

        # ── Desired wrench ──
        W_des = self._compute_desired_wrench()

        # ── Relative foot positions ──
        r_dict = {leg: self._ctrls[leg].get_foot_pos() - pos for leg in stance}

        # ── Solve ──
        if n == 2:
            f_vec = self._solve_6x6(W_des, r_dict, stance)
        else:
            f_vec = self._solve_lsq(W_des, r_dict, stance)

        # ── Friction enforcement ──
        f_vec, feasible, mu_used = self._enforce_friction(
            f_vec, W_des, r_dict, stance
        )
        self._momentum_feasible = feasible
        self._momentum_mu_max = mu_used

        # ── Negate: ground→foot → foot→ground ──
        f_vec = -f_vec

        # ── Clip vertical ──
        for i in range(n):
            idx = 3 * i
            f_vec[idx + 2] = np.clip(f_vec[idx + 2], -self._max_fz, 0.0)

        # ── Assemble ──
        foot_forces = {}
        for i, leg in enumerate(stance):
            idx = 3 * i
            foot_forces[leg] = f_vec[idx:idx + 3].copy()

        return foot_forces

    # ── Control loop ──────────────────────────────────────────────────────

    def control(self, t: float):
        """Momentum-based control: kinematic placement → 6×6 solve → impedance."""
        _, _, vel, angvel, _, _, yaw = self._body_state()

        # Yaw angle tracking (P+I, world-frame wz from quaternion)
        if self._initial_yaw is None:
            self._initial_yaw = yaw
            self._prev_yaw = yaw
            self._prev_t = t
        desired_yaw = self._initial_yaw + self.target_vyaw * t
        yaw_angle_error = self._wrap_pi(desired_yaw - yaw)
        dt_yaw = t - self._prev_t
        if dt_yaw > 1e-9:
            self._world_wz = self._wrap_pi(yaw - self._prev_yaw) / dt_yaw
        self._prev_yaw = yaw
        self._prev_t = t
        lat_ratio = abs(self.target_vy) / max(abs(self.target_vx) + abs(self.target_vy), 0.01)
        self._yaw_error_integral += yaw_angle_error * 0.002 * lat_ratio
        self._yaw_error_integral = float(np.clip(self._yaw_error_integral, -1.0, 1.0))

        dvx = self.target_vx - vel[0]
        dvy = self.target_vy - vel[1]
        dwz = self.target_vyaw - angvel[2]  # qvel rate damping (not _world_wz)

        # Kinematic foot placement (Raibert-style) with yaw angle P+I
        step_delta = np.clip(self._planner.K_kin_vx * dvx, -0.15, 0.15)
        lat_offset = np.clip(self._planner.K_kin_vy * dvy, -0.05, 0.05)
        yaw_offset = np.clip(
            self._planner.K_kin_wz * dwz
            + self._K_yaw_angle_kin * yaw_angle_error
            + 2.0 * self._yaw_error_integral,
            -0.5, 0.5)
        self._planner.set_kinematic_adjustments(step_delta, lat_offset, yaw_offset)
        self._planner.target_vyaw = 0.0
        self._yaw_angle_error = yaw_angle_error

        # Momentum-based force distribution
        foot_forces = self._compute_momentum_forces(t)

        # Impedance control (inherited)
        self._apply_leg_impedance(t, foot_forces)

        # Direct yaw torque on freejoint
        yaw_err = getattr(self, '_yaw_angle_error', 0.0)
        yaw_int = getattr(self, '_yaw_error_integral', 0.0)
        Mz_direct = (15.0 * yaw_err + 20.0 * yaw_int)
        trunk_id = self._sim._body_ids.get("trunk")
        if trunk_id is not None:
            jnt_id = self._sim._model.body_jntadr[trunk_id]
            dof_adr = self._sim._model.jnt_dofadr[jnt_id]
            self._sim._data.qfrc_applied[dof_adr + 5] += float(np.clip(
                Mz_direct, -10.0, 10.0))

        # Stats for Tier-1 adaptation
        self.record_stats()

        # ── Record velocity tracking metrics ──
        if self._metrics_enabled:
            pos, _, vel, angvel, roll, pitch, _ = self._body_state()
            self._metrics.record(t, vel, angvel, roll, pitch, pos[2],
                                self.target_vx, self.target_vy, self.target_vyaw)

        if self._adapt_params:
            self._adapt_counter += 1
            if self._adapt_counter >= self.ADAPT_INTERVAL:
                self._adapt_counter = 0
                self._adapt_gait_params()

    # ── Statistics ────────────────────────────────────────────────────────

    def record_stats(self):
        """Record momentum friction utilisation and velocity error."""
        _, _, vel, _, _, _, _ = self._body_state()
        self._mu_history.append(self._momentum_mu_max)
        self._vx_error_history.append(self.target_vx - vel[0])
        if len(self._mu_history) > 200:
            self._mu_history = self._mu_history[-100:]
            self._vx_error_history = self._vx_error_history[-100:]

    @property
    def friction_stats(self) -> dict:
        """Return momentum-controller statistics."""
        return {
            "mu_max_used": self._momentum_mu_max,
            "feasible": self._momentum_feasible,
            "cond_A": self._momentum_cond,
            "n_stance": self._momentum_n_stance,
            "T_cycle": self._scheduler.T_cycle,
            "step_length": self._scheduler.params.step_length,
        }
