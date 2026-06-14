"""SRB (Single Rigid Body) Convex MPC for quadruped locomotion.

Based on the MIT Cheetah approach (Di Carlo et al., IROS 2018):
  - Linearized SRB dynamics (small roll/pitch, ignore gyroscopic terms)
  - Convex QP with linearized friction cone constraints
  - Compact form: eliminates states, optimizes only ground reaction forces

The solver outputs desired GRFs that feed into the MIT impedance controller
as feed-forward forces: τ = Jᵀ·(F_mpc + Kp·Δx + Kd·Δv).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sparse
import osqp

from .gait import ALL_LEGS, GaitScheduler


class SrbMpcSolver:
    """Convex MPC solver for SRB quadruped dynamics.

    State x ∈ R¹²: [roll, pitch, yaw, px, py, pz, ωx, ωy, ωz, vx, vy, vz]
    Control u ∈ R¹²: [f_FR, f_FL, f_RR, f_RL]  (3D forces per foot in world frame)

    Prediction horizon: N steps at dt_mpc each.
    QP size: 12·N variables, ~5·(n_stance)·N linear constraints.
    """

    def __init__(self, mass: float, inertia_body: np.ndarray,
                 hip_offsets_com: Dict[str, np.ndarray],
                 N: int = 10, dt_mpc: float = 0.03,
                 mu: float = 0.6, fz_min: float = 10.0, fz_max: float = 150.0):
        """Initialize the SRB MPC solver.

        Args:
            mass: Total robot mass (kg).
            inertia_body: Body-frame inertia diagonal [Ixx, Iyy, Izz] (kg·m²).
            hip_offsets_com: Hip positions relative to CoM in body frame, keyed by leg.
            N: Prediction horizon steps (default 10).
            dt_mpc: MPC time step (s), typically 0.03 → 0.3s horizon.
            mu: Friction coefficient (default 0.6).
            fz_min: Minimum vertical force per foot (N), ensures contact.
            fz_max: Maximum vertical force per foot (N).
        """
        self.mass = mass
        self.I_body_diag = np.asarray(inertia_body, dtype=float)
        self.I_body_inv = np.diag(1.0 / self.I_body_diag)
        self.hip_body = {leg: np.asarray(p, dtype=float)
                         for leg, p in hip_offsets_com.items()}
        self.N = N
        self.dt = dt_mpc
        self.mu = mu
        self.fz_min = fz_min
        self.fz_max = fz_max

        self.nx = 12   # state dimension
        self.nu = 12   # control dimension (4 legs × 3D force)

        # Full state gravity vector: only affects v̇_z (index 11)
        self._g_vec = np.zeros(self.nx)
        self._g_vec[11] = -9.81

        # ── Build constant A_c matrix ──
        self._A_c = self._build_A_c()

        # ── Pre-compute A_d (constant, A_c² = 0) ──
        self._A_d = np.eye(self.nx) + self._A_c * self.dt

        # ── Pre-compute gravity offset ──
        self._g_d = (np.eye(self.nx) * self.dt
                     + self._A_c * (self.dt**2 / 2)) @ self._g_vec

        # ── Pre-compute gravity accumulation over horizon ──
        self._X_gravity = self._build_X_gravity()

        # ── Pre-compute sparse constraint matrix C (values are constant!) ──
        self._C, self._constraint_per_leg = self._build_constraint_matrix()

        # ── Cost weights ──
        # State cost: [roll, pitch, yaw, px, py, pz, ωx, ωy, ωz, vx, vy, vz]
        # State cost weights: focus MPC on what it's good at (velocity/height),
        # leave orientation stabilization to the impedance layer.
        self.Q_diag = np.array([
            0.1,     # roll  — impedance layer handles this
            0.1,     # pitch — impedance layer handles this
            1.0,     # yaw   — track yaw (SRB limited, accept limitation)
            1.0,     # px    — weak position tracking
            1.0,     # py
            500.0,   # pz    — height tracking
            0.1,     # ωx    — impedance layer handles
            0.1,     # ωy    — impedance layer handles
            50.0,    # ωz    — yaw rate damping
            20000.0, # vx    — forward velocity: primary task
            8000.0,  # vy    — lateral velocity
            200.0,   # vz    — damp vertical oscillation
        ])

        # Control cost per leg (3D force) — mild regularization to smooth forces
        self.R_diag = np.array([1e-3, 1e-3, 1e-4])

        # ── OSQP problem (created on first solve) ──
        self._prob: Optional[osqp.OSQP] = None
        self._last_contact_mask = None

    # ═════════════════════════════════════════════════════════════════════════
    # A_c matrix (constant)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_A_c(self) -> np.ndarray:
        """Build the continuous-time A matrix.

        Under small-angle assumption:
          Θ̇ = ω          (Euler angle rates ≈ angular velocity)
          ṗ  = v          (position derivative = velocity)
          ω̇ = 0           (in A_c; angular acceleration from forces is in B_c·u)
          v̇ = 0           (in A_c; linear acceleration from forces is in B_c·u)

        Returns:
            A_c ∈ R¹²ˣ¹², nilpotent (A_c² = 0).
        """
        A = np.zeros((self.nx, self.nx))
        # Θ̇ = ω:  rows 0-2, cols 6-8
        A[0:3, 6:9] = np.eye(3)
        # ṗ = v:   rows 3-5, cols 9-11
        A[3:6, 9:12] = np.eye(3)
        # ω̇ and v̇ rows are all zeros → A_c² = 0
        return A

    # ═════════════════════════════════════════════════════════════════════════
    # B_c matrix (depends on foot positions relative to CoM)
    # ═════════════════════════════════════════════════════════════════════════

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Cross-product skew-symmetric matrix [v]×."""
        return np.array([
            [0.0,    -v[2],   v[1]],
            [v[2],    0.0,   -v[0]],
            [-v[1],   v[0],   0.0],
        ])

    def _rotate_inertia_inv(self, yaw: float) -> np.ndarray:
        """Compute I_world^{-1} = R_z(yaw) · I_body^{-1} · R_z(yaw)^T."""
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ self.I_body_inv @ Rz.T

    def build_B_c(self, foot_positions: Dict[str, np.ndarray],
                  com_position: np.ndarray, yaw: float) -> np.ndarray:
        """Build B_c from current foot positions in world frame.

        Args:
            foot_positions: Dict leg → foot position [x, y, z] in WORLD frame.
            com_position: CoM position [x, y, z] in world frame.
            yaw: Current yaw angle (rad).

        Returns:
            B_c ∈ R¹²ˣ¹².
        """
        B = np.zeros((self.nx, self.nu))
        I_inv_world = self._rotate_inertia_inv(yaw)

        for i, leg in enumerate(ALL_LEGS):
            r = foot_positions[leg] - com_position  # vector from CoM to foot
            col_start = i * 3  # 3 force components per leg

            # Angular acceleration: ω̇ = I⁻¹ · [r]× · f
            B[6:9, col_start:col_start + 3] = I_inv_world @ self._skew(r)

            # Linear acceleration: v̇ = (1/m) · I · f
            B[9:12, col_start:col_start + 3] = (1.0 / self.mass) * np.eye(3)

        return B

    def build_B_d(self, B_c: np.ndarray) -> np.ndarray:
        """Discretize B: B_d = (I·dt + A_c·dt²/2) · B_c."""
        M = np.eye(self.nx) * self.dt + self._A_c * (self.dt**2 / 2.0)
        return M @ B_c

    # ═════════════════════════════════════════════════════════════════════════
    # Horizon propagation matrices
    # ═════════════════════════════════════════════════════════════════════════

    def _A_d_power(self, k: int) -> np.ndarray:
        """Compute A_d^k = I + k·A_c·dt  (exact since A_c² = 0)."""
        return np.eye(self.nx) + k * self._A_c * self.dt

    def build_horizon_matrices(self, B_d: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Build A_qp and B_qp for the compact QP form: X = A_qp·x0 + B_qp·U.

        Args:
            B_d: Discrete B matrix at current time step (12×12).

        Returns:
            A_qp ∈ R^{12N × 12}, B_qp ∈ R^{12N × 12N}.
        """
        N = self.N
        A_qp = np.zeros((self.nx * N, self.nx))
        B_qp = np.zeros((self.nx * N, self.nu * N))

        for k in range(N):
            row_slice = slice(k * self.nx, (k + 1) * self.nx)

            # A_qp: row k is A_d^{k+1}
            A_qp[row_slice, :] = self._A_d_power(k + 1)

            # B_qp: block (k, j) = A_d^{k-j} · B_d  for j ≤ k
            for j in range(k + 1):
                col_slice = slice(j * self.nu, (j + 1) * self.nu)
                B_qp[row_slice, col_slice] = (self._A_d_power(k - j) @ B_d)

        return A_qp, B_qp

    def _build_X_gravity(self) -> np.ndarray:
        """Pre-compute gravity contribution over the horizon.

        X_gravity[k] = Σ_{j=0}^{k} A_d^{j} · g_d

        Since A_d^{j} = I + j·A_c·dt:
        Σ_{j=0}^{k} (I + j·A_c·dt) = (k+1)·I + k(k+1)/2·A_c·dt
        """
        N = self.N
        X_g = np.zeros(self.nx * N)
        sum_A_powers = np.zeros((self.nx, self.nx))
        for k in range(N):
            # sum_{j=0}^{k} A_d^j
            sum_mat = ((k + 1) * np.eye(self.nx)
                       + (k * (k + 1) / 2.0) * self._A_c * self.dt)
            X_g[k * self.nx:(k + 1) * self.nx] = sum_mat @ self._g_d
        return X_g

    # ═════════════════════════════════════════════════════════════════════════
    # Constraint matrix (sparse, constant values)
    # ═════════════════════════════════════════════════════════════════════════

    def _build_constraint_matrix(self
                                 ) -> Tuple[sparse.csc_matrix, int]:
        """Build the sparse constraint matrix C.

        Per leg per timestep: 9 rows
          Rows 0-3: friction cone (4 edges: ±fx≤μfz, ±fy≤μfz) — stance only
          Row 4: fz ≥ fz_min — stance; disabled for swing
          Row 5: fz ≤ fz_max — stance; disabled for swing
          Row 6: fz = 0 — swing equality (l=u=0); disabled for stance
          Row 7: fx = 0 — swing equality; disabled for stance
          Row 8: fy = 0 — swing equality; disabled for stance

        Swing legs: rows 6-8 set l=u=0 → fx=fy=fz=0 enforced in QP
        """
        mu = self.mu
        N = self.N
        n_per_leg = 9  # 4 friction + 2 fz bounds + 3 zero-force eq
        n_constraints = N * 4 * n_per_leg
        n_vars = self.nu * N

        rows = []
        cols = []
        data = []

        for k in range(N):
            for leg_idx in range(4):
                base_row = (k * 4 + leg_idx) * n_per_leg
                base_col = (k * self.nu) + leg_idx * 3
                fx_c, fy_c, fz_c = base_col, base_col + 1, base_col + 2

                # 0-3: Friction cone (μ·fz ≥ ±fx, μ·fz ≥ ±fy)
                rows.extend([base_row+0, base_row+0]); cols.extend([fx_c, fz_c]); data.extend([-1.0, mu])
                rows.extend([base_row+1, base_row+1]); cols.extend([fx_c, fz_c]); data.extend([1.0, mu])
                rows.extend([base_row+2, base_row+2]); cols.extend([fy_c, fz_c]); data.extend([-1.0, mu])
                rows.extend([base_row+3, base_row+3]); cols.extend([fy_c, fz_c]); data.extend([1.0, mu])

                # 4: fz ≥ fz_min
                rows.append(base_row + 4); cols.append(fz_c); data.append(1.0)
                # 5: fz ≤ fz_max
                rows.append(base_row + 5); cols.append(fz_c); data.append(1.0)
                # 6-8: fx=0, fy=0, fz=0 (swing equality)
                rows.append(base_row + 6); cols.append(fz_c); data.append(1.0)
                rows.append(base_row + 7); cols.append(fx_c); data.append(1.0)
                rows.append(base_row + 8); cols.append(fy_c); data.append(1.0)

        C = sparse.coo_matrix((data, (rows, cols)),
                              shape=(n_constraints, n_vars)).tocsc()
        return C, n_per_leg

    # ═════════════════════════════════════════════════════════════════════════
    # Contact schedule
    # ═════════════════════════════════════════════════════════════════════════

    def compute_contact_schedule(self, t: float, scheduler: GaitScheduler
                                 ) -> np.ndarray:
        """Compute boolean contact mask over the MPC horizon.

        Returns:
            contact (4 × N): contact[leg_idx, k] = True if leg is in stance
            at MPC step k (starting from current time t).
        """
        contact = np.zeros((4, self.N), dtype=bool)
        for k in range(self.N):
            t_k = t + k * self.dt
            stance_legs = scheduler.get_stance_legs(t_k)
            for li, leg in enumerate(ALL_LEGS):
                contact[li, k] = (leg in stance_legs)
        return contact

    # ═════════════════════════════════════════════════════════════════════════
    # Reference trajectory
    # ═════════════════════════════════════════════════════════════════════════

    def build_reference(self, x0: np.ndarray,
                        target_vx: float = 0.3,
                        target_vy: float = 0.0,
                        target_vyaw: float = 0.0,
                        target_height: float = 0.28
                        ) -> np.ndarray:
        """Build reference state trajectory over the horizon.

        Uses zero-order hold: reference is constant at target values.
        Orientation ref is zero (keep body level), velocity ref is target.

        Returns:
            X_ref ∈ R^{12·N}.
        """
        x_ref_step = np.zeros(self.nx)
        # roll, pitch → 0 (keep level)
        x_ref_step[0] = 0.0  # roll ref
        x_ref_step[1] = 0.0  # pitch ref
        x_ref_step[2] = x0[2]  # yaw ref: maintain current yaw
        # position: don't track absolute pos, just height
        x_ref_step[3] = 0.0   # px (don't care)
        x_ref_step[4] = 0.0   # py (don't care)
        x_ref_step[5] = target_height
        # angular velocity: damp to 0
        x_ref_step[6] = 0.0
        x_ref_step[7] = 0.0
        x_ref_step[8] = target_vyaw  # yaw rate target
        # linear velocity: track target
        x_ref_step[9] = target_vx
        x_ref_step[10] = target_vy
        x_ref_step[11] = 0.0  # vz → 0

        X_ref = np.tile(x_ref_step, self.N)
        return X_ref

    # ═════════════════════════════════════════════════════════════════════════
    # QP construction and solve
    # ═════════════════════════════════════════════════════════════════════════

    def _build_cost_matrices(self, A_qp: np.ndarray, B_qp: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Build H and g for the compact QP.

        H = 2 · (B_qp^T · Q_bar · B_qp + R_bar)
        g = 2 · B_qp^T · Q_bar · (A_qp · x0 + X_gravity - X_ref)

        Note: H and g are built without x0/X_ref for g; caller adds the
        x0/X_ref dependent part later.
        """
        N = self.N
        n_vars = self.nu * N

        # Build block-diagonal Q_bar and R_bar
        Q_bar = sparse.block_diag([np.diag(self.Q_diag)] * N, format='csc')
        R_bar = sparse.block_diag([np.diag(self.R_diag)] * N * 4, format='csc')

        # H: use dense for B_qp^T @ Q_bar @ B_qp, then add sparse R
        # Q_bar is diagonal → Q_bar @ B_qp is element-wise
        B_qp_sparse = sparse.csc_matrix(B_qp)
        H = B_qp_sparse.T @ Q_bar @ B_qp_sparse
        H = H + R_bar
        H = 2.0 * H

        return H, B_qp_sparse, Q_bar

    def solve(self, x0: np.ndarray, X_ref: np.ndarray,
              foot_positions: Dict[str, np.ndarray],
              com_position: np.ndarray,
              yaw: float, contact: np.ndarray
              ) -> Optional[np.ndarray]:
        """Solve the MPC QP and return the first-step GRFs.

        Args:
            x0: Current state [roll,pitch,yaw, px,py,pz, ωx,ωy,ωz, vx,vy,vz].
            X_ref: Reference trajectory over horizon (12·N,).
            foot_positions: Dict leg → foot position [x,y,z] in WORLD frame.
            com_position: CoM position in world frame.
            yaw: Current yaw angle.
            contact: Boolean mask (4 × N), True for stance.

        Returns:
            u_first ∈ R¹² (GRFs for all 4 legs in world frame, swing legs ≈ 0),
            or None if QP fails.
        """
        # ── Build dynamics matrices ──
        B_c = self.build_B_c(foot_positions, com_position, yaw)
        B_d = self.build_B_d(B_c)
        A_qp, B_qp = self.build_horizon_matrices(B_d)

        # ── Build cost ──
        H, B_qp_sparse, Q_bar = self._build_cost_matrices(A_qp, B_qp)

        # Linear term: g = 2 · B_qp^T · Q_bar · (A_qp @ x0 + X_gravity - X_ref)
        err = A_qp @ x0 + self._X_gravity - X_ref
        g = 2.0 * B_qp_sparse.T @ (Q_bar @ err)
        g = np.asarray(g).flatten()

        # ── Build constraint bounds l, u ──
        l, u = self._build_constraint_bounds(contact)

        # ── Convert H to sparse for OSQP ──
        P_sparse = sparse.csc_matrix(H)

        # ── Setup or update OSQP ──
        n_constraints = len(l)
        if self._prob is None:
            # First solve: full setup
            self._prob = osqp.OSQP()
            self._prob.setup(P=P_sparse, q=g, A=self._C, l=l, u=u,
                            warm_start=True, verbose=False,
                            eps_abs=1e-4, eps_rel=1e-4,
                            max_iter=2000, polish=False)
        else:
            # Reuse solver object: OSQP preserves internal state for warm-start.
            # The constraint matrix (self._C) is constant; only l, u, and q change.
            # P changes slightly due to foot movement, but setup() with
            # warm_start=True initializes from the previous solution.
            self._prob.setup(P=P_sparse, q=g, A=self._C, l=l, u=u,
                            warm_start=True, verbose=False,
                            eps_abs=1e-4, eps_rel=1e-4,
                            max_iter=2000, polish=False)

        self._last_contact_mask = contact.copy()

        # ── Solve ──
        result = self._prob.solve()

        if result.info.status_val not in (1, 2):  # solved or solved inaccurate
            return None

        U = result.x
        if U is None:
            return None

        # Return first-step forces: 12D vector [f_FR, f_FL, f_RR, f_RL]
        u_first = U[:self.nu].copy()

        # Post-process: force swing legs to exactly zero
        for leg_idx in range(4):
            if not contact[leg_idx, 0]:  # swing at first step
                u_first[leg_idx * 3:(leg_idx + 1) * 3] = 0.0

        # Clamp safety
        for leg_idx in range(4):
            fz = u_first[leg_idx * 3 + 2]
            if fz < 0.0:
                u_first[leg_idx * 3 + 2] = 0.0
            elif fz > self.fz_max:
                u_first[leg_idx * 3] *= self.fz_max / fz
                u_first[leg_idx * 3 + 1] *= self.fz_max / fz
                u_first[leg_idx * 3 + 2] = self.fz_max

        return u_first

    def _contact_changed(self, contact: np.ndarray) -> bool:
        """Check if contact schedule changed since last solve."""
        if self._last_contact_mask is None:
            return True
        return not np.array_equal(contact, self._last_contact_mask)

    def _build_constraint_bounds(self, contact: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Build l, u vectors for the constraints based on contact mask.

        For stance legs:
          - Friction edges: l=0, u=∞
          - fz_min: l=fz_min, u=∞  (row 4)
          - fz_max: l=-∞, u=fz_max  (row 5)
          - fz_eq: l=-∞, u=∞  (row 6, unused)

        For swing legs:
          - Friction edges: l=-∞, u=∞  (disabled)
          - fz_min: l=-∞, u=∞  (disabled)
          - fz_max: l=-∞, u=∞  (disabled)
          - fz_eq: l=0, u=0  (force fz=0)
          - Additionally, we need fx=0, fy=0. We handle these via implicit
            l=u=0 constraints but our constraint matrix only has fz rows.
            → We'll zero swing forces in post-processing instead.
        """
        N = self.N
        n_per_leg = self._constraint_per_leg
        n_constraints = N * 4 * n_per_leg

        l = np.full(n_constraints, -np.inf)
        u = np.full(n_constraints, np.inf)

        for k in range(N):
            for leg_idx in range(4):
                base_row = (k * 4 + leg_idx) * n_per_leg

                if contact[leg_idx, k]:
                    # Stance: active friction + force bounds
                    l[base_row + 0] = 0.0     # μfz - fx ≥ 0
                    l[base_row + 1] = 0.0     # μfz + fx ≥ 0
                    l[base_row + 2] = 0.0     # μfz - fy ≥ 0
                    l[base_row + 3] = 0.0     # μfz + fy ≥ 0
                    l[base_row + 4] = self.fz_min  # fz ≥ fz_min
                    u[base_row + 5] = self.fz_max  # fz ≤ fz_max
                    # rows 6-8: disabled (not swing)
                else:
                    # Swing: disable friction, zero forces
                    l[base_row + 0] = -np.inf; u[base_row + 0] = np.inf  # disable
                    l[base_row + 1] = -np.inf; u[base_row + 1] = np.inf  # disable
                    l[base_row + 2] = -np.inf; u[base_row + 2] = np.inf  # disable
                    l[base_row + 3] = -np.inf; u[base_row + 3] = np.inf  # disable
                    l[base_row + 4] = -np.inf; u[base_row + 4] = np.inf  # disable
                    l[base_row + 5] = -np.inf; u[base_row + 5] = np.inf  # disable
                    # rows 6-8: force fx=fy=fz=0
                    l[base_row + 6] = 0.0; u[base_row + 6] = 0.0  # fz=0
                    l[base_row + 7] = 0.0; u[base_row + 7] = 0.0  # fx=0
                    l[base_row + 8] = 0.0; u[base_row + 8] = 0.0  # fy=0

        return l, u

    def forces_to_dict(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat force vector to dict keyed by leg name."""
        result = {}
        for i, leg in enumerate(ALL_LEGS):
            result[leg] = u[i * 3:(i + 1) * 3].copy()
        return result
