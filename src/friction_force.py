"""Static-friction-constrained force distribution for quadruped stance legs.

Core idea (user's insight):
  For a trot gait with 2 stance legs, each leg has fx, fy (4 variables total)
  while the body needs Fx, Fy, Mz (3 equations). The system is underdetermined
  with a 1-dimensional null space.

  We solve:
    min ||f||²  subject to  A·f = b,  |fxᵢ| ≤ μ·fzᵢ,  |fyᵢ| ≤ μ·fzᵢ

  The solution is f = f_particular + α·f_null, where α is chosen to satisfy
  the linearised friction cone constraints with minimum norm.
"""

from typing import Dict, List, Optional

import numpy as np


class FrictionForceDistributor:
    """Distribute desired body wrench to stance legs under friction constraints.

    For N stance legs, there are 2N variables (fx, fy per leg) and 3 equations
    (Fx, Fy, Mz). The system is:
      - Underdetermined for N=1 (2 vars, 3 eqs — rank ≤ 2) → best-effort LS
      - Underdetermined for N=2 (4 vars, 3 eqs) → 1D null space  ← primary case
      - Determined/overdetermined for N≥3 → damped least squares

    The linearised friction cone (4-sided pyramid) is used as constraint:
      |fxᵢ| ≤ μ·fzᵢ,   |fyᵢ| ≤ μ·fzᵢ

    Attributes:
        mu_max: Maximum static friction coefficient (μₛ).
        mu_utilized: Friction utilisation ratio from the last solve,
                     defined as max_i(||f_horiz_i|| / (mu_max * fz_i)).
                     Used by the Tier-1 gait-parameter optimiser.
    """

    def __init__(self, mu_max: float = 0.6):
        if mu_max <= 0:
            raise ValueError(f"mu_max must be positive, got {mu_max}")
        self.mu_max = mu_max
        self._mu_utilized: float = 0.0
        self._last_alpha: float = 0.0
        self._feasible: bool = True

    @property
    def mu_utilized(self) -> float:
        """Friction utilisation from the last call to distribute()."""
        return self._mu_utilized

    @property
    def last_alpha(self) -> float:
        """Null-space parameter α from the last 2-leg solve."""
        return self._last_alpha

    @property
    def feasible(self) -> bool:
        """Whether the last solve was fully feasible (α=0 satisfied constraints)."""
        return self._feasible

    # ── Public API ──────────────────────────────────────────────────────────

    def distribute(
        self,
        desired_wrench: np.ndarray,      # [Fx, Fy, Mz]
        foot_positions: Dict[str, np.ndarray],  # world-frame foot pos per leg
        foot_fz: Dict[str, float],       # vertical force (N) per leg
        stance_legs: List[str],
        com_position: np.ndarray,        # [x, y, z]
    ) -> Dict[str, np.ndarray]:
        """Distribute desired body wrench to horizontal forces on stance legs.

        Args:
            desired_wrench: [Fx, Fy, Mz] in world frame (N, N, N·m).
            foot_positions: {leg: [x, y, z]} world-frame foot positions.
            foot_fz: {leg: fz} vertical force already allocated to each leg.
            stance_legs: List of leg names currently in stance.
            com_position: [x, y, z] centre-of-mass position.

        Returns:
            Dict mapping each stance leg → [fx, fy] horizontal forces (N).
        """
        n = len(stance_legs)
        if n == 0:
            return {}

        b = np.asarray(desired_wrench, dtype=float)

        # Build relative position vectors
        r = {}
        for leg in stance_legs:
            r[leg] = np.asarray(foot_positions[leg], dtype=float) - com_position

        if n == 1:
            return self._distribute_single(b, r, foot_fz, stance_legs)
        elif n == 2:
            return self._distribute_nullspace(b, r, foot_fz, stance_legs)
        else:
            return self._distribute_lsq(b, r, foot_fz, stance_legs)

    # ── Single-leg best-effort ──────────────────────────────────────────────

    def _distribute_single(self, b, r, fz, legs):
        """Best-effort: project wrench onto the single leg's friction cone."""
        leg = legs[0]
        r_vec = r[leg]
        mu = self.mu_max
        fz_i = max(abs(fz[leg]), 1.0)

        # Build 3×2 Jacobian for this leg
        J = np.zeros((3, 2))
        J[0, 0] = 1.0           # fx → Fx
        J[1, 1] = 1.0           # fy → Fy
        J[2, 0] = -r_vec[1]     # fx → Mz
        J[2, 1] = r_vec[0]      # fy → Mz

        # Minimum-norm least squares
        fh = np.linalg.lstsq(J, b, rcond=None)[0]

        # Clip to friction cone
        max_fh = mu * fz_i
        f_norm = np.linalg.norm(fh)
        if f_norm > max_fh:
            fh *= max_fh / f_norm

        self._mu_utilized = f_norm / (mu * fz_i) if mu * fz_i > 0 else 0.0
        self._feasible = (f_norm <= max_fh)
        self._last_alpha = 0.0
        return {leg: fh}

    # ── 2-leg null-space method (primary case for trot / pace) ──────────────

    def _distribute_nullspace(self, b, r, fz, legs):
        """Solve the underdetermined 3×4 system via null-space parameterisation.

        f_horiz = f_particular + α · f_null

        α is chosen to MINIMISE the maximum friction utilisation ratio
        max_i(||fh_i|| / (μ·fz_i)) — i.e. the solution farthest from slipping.

        When the feasible α-range is empty, the wrench is scaled down until
        a feasible solution exists.
        """
        L, R = legs[0], legs[1]
        rL, rR = r[L], r[R]
        mu = self.mu_max
        fzL = max(abs(fz[L]), 1.0)
        fzR = max(abs(fz[R]), 1.0)

        # ── Build A (3×4) and solve particular + null space ──
        A = np.array([
            [1.0, 0.0,  1.0, 0.0],
            [0.0, 1.0,  0.0, 1.0],
            [-rL[1], rL[0], -rR[1], rR[0]],
        ])

        # Particular solution: minimum-norm via pseudoinverse
        A_pinv = np.linalg.pinv(A)
        f_p = A_pinv @ b

        # Null space: last right singular vector (4D → 3 eqns → 1D kernel)
        try:
            _, s, Vt = np.linalg.svd(A, full_matrices=False)
            null_tol = 1e-10
            null_vecs = Vt[s < null_tol] if s[-1] < null_tol else Vt[-1:]
        except np.linalg.LinAlgError:
            null_vecs = np.zeros((1, 4))

        if null_vecs.shape[0] == 0 or np.allclose(null_vecs[0], 0):
            # Degenerate: no null space (co-linear feet w.r.t. CoM)
            fh = np.array([f_p[0], f_p[1], f_p[2], f_p[3]])
            fh = self._clip_horizontal(fh, mu, fzL, fzR)
            self._mu_utilized = self._compute_mu_utilized(fh, mu, fzL, fzR)
            self._feasible = True
            self._last_alpha = 0.0
            return {
                L: fh[0:2].copy(),
                R: fh[2:4].copy(),
            }

        n_vec = null_vecs[0]  # 4D null-space basis vector

        # ── Precompute utilisation coefficients for fast evaluation ──
        # μ_i(α) = sqrt(A_i·α² + B_i·α + C_i) / (μ·fz_i)
        def _util_coeffs(offset, fz_i):
            fx, fy = f_p[offset], f_p[offset + 1]
            nx, ny = n_vec[offset], n_vec[offset + 1]
            A_i = nx * nx + ny * ny
            B_i = 2.0 * (fx * nx + fy * ny)
            C_i = fx * fx + fy * fy
            denom = mu * fz_i
            return A_i, B_i, C_i, denom

        AL, BL, CL, dL = _util_coeffs(0, fzL)
        AR, BR, CR, dR = _util_coeffs(2, fzR)

        def _mu_max_at(alpha: float) -> float:
            """max(μ_L, μ_R) at given α."""
            muL = np.sqrt(max(0.0, AL * alpha * alpha + BL * alpha + CL)) / dL
            muR = np.sqrt(max(0.0, AR * alpha * alpha + BR * alpha + CR)) / dR
            return max(muL, muR)

        # ── Compute feasible α range from friction constraints ──
        # For each leg i and axis c, constraint: |f_i^c| ≤ μ·fz_i
        alpha_lower = -np.inf
        alpha_upper = np.inf

        for i, (fz_i, offset) in enumerate([(fzL, 0), (fzR, 2)]):
            for axis_idx in range(2):
                f_p_i = f_p[offset + axis_idx]
                n_i = n_vec[offset + axis_idx]
                max_f = mu * fz_i

                if abs(n_i) > 1e-15:
                    # f_p_i + α·n_i ≤ max_f  →  α·n_i ≤ max_f - f_p_i
                    if n_i > 0:
                        alpha_upper = min(alpha_upper, (max_f - f_p_i) / n_i)
                    else:
                        alpha_lower = max(alpha_lower, (max_f - f_p_i) / n_i)
                    # -(f_p_i + α·n_i) ≤ max_f  →  -α·n_i ≤ max_f + f_p_i
                    if -n_i > 0:
                        alpha_upper = min(alpha_upper, (max_f + f_p_i) / (-n_i))
                    else:
                        alpha_lower = max(alpha_lower, (max_f + f_p_i) / (-n_i))
                elif abs(f_p_i) > max_f + 1e-12:
                    alpha_lower = np.inf
                    alpha_upper = -np.inf

        # ── Choose α: minimise max(μ_L, μ_R) in feasible range ──
        if alpha_lower <= alpha_upper:
            self._feasible = True
            # Golden-section search for min of convex max(μ_L, μ_R)
            alpha = self._golden_section_min(_mu_max_at, alpha_lower, alpha_upper)
        else:
            # Infeasible: use α that minimises max utilisation, then scale
            self._feasible = False
            # Search over a reasonable range around the boundaries
            lo = min(alpha_lower, alpha_upper) - 10.0
            hi = max(alpha_lower, alpha_upper) + 10.0
            alpha = self._golden_section_min(_mu_max_at, lo, hi)

        self._last_alpha = alpha
        fh = f_p + alpha * n_vec

        # ── Safety clip (numerical rounding) ──
        fh = self._clip_horizontal(fh, mu, fzL, fzR)

        # ── If infeasible, scale wrench down to feasibility ──
        if not self._feasible:
            fh = self._scale_wrench_to_feasible(A, b, fh, mu, fzL, fzR)

        # ── Compute utilisation ──
        self._mu_utilized = self._compute_mu_utilized(fh, mu, fzL, fzR)

        return {
            L: fh[0:2].copy(),
            R: fh[2:4].copy(),
        }

    # ── Golden-section search for 1D convex min ──────────────────────────

    @staticmethod
    def _golden_section_min(f, a: float, b: float, tol: float = 1e-8,
                            max_iter: int = 80) -> float:
        """Find α ∈ [a, b] that minimises convex function f(α)."""
        phi = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ≈ 0.618
        lo, hi = float(a), float(b)
        if lo > hi:
            lo, hi = hi, lo
        if hi - lo < 1e-12:
            return (lo + hi) / 2.0

        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        fc, fd = f(c), f(d)

        for _ in range(max_iter):
            if hi - lo < tol * (abs(lo) + abs(hi) + 1e-12):
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = f(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = f(d)

        return (lo + hi) / 2.0

    # ── 3+ leg damped least squares ─────────────────────────────────────────

    def _distribute_lsq(self, b, r, fz, legs):
        """Damped least squares with per-leg friction bounding for N ≥ 3."""
        n = len(legs)
        mu = self.mu_max

        # Build 3×2N Jacobian
        J = np.zeros((3, 2 * n))
        for i, leg in enumerate(legs):
            J[0, 2 * i] = 1.0
            J[1, 2 * i + 1] = 1.0
            rx, ry = r[leg][0], r[leg][1]
            J[2, 2 * i] = -ry
            J[2, 2 * i + 1] = rx

        # Damped least squares
        damping = 1e-3 * np.eye(2 * n)
        fh = np.linalg.solve(J.T @ J + damping, J.T @ b)

        # Per-leg friction clipping
        max_util = 0.0
        for i, leg in enumerate(legs):
            f_i = fh[2 * i:2 * i + 2]
            max_f = mu * max(abs(fz[leg]), 1.0)
            f_norm = np.linalg.norm(f_i)
            if f_norm > max_f:
                f_i *= max_f / f_norm
                fh[2 * i:2 * i + 2] = f_i
            max_util = max(max_util, f_norm / max_f if max_f > 0 else 0.0)

        self._mu_utilized = max_util
        self._feasible = (max_util <= 1.0 + 1e-6)
        self._last_alpha = 0.0

        return {leg: fh[2 * i:2 * i + 2].copy() for i, leg in enumerate(legs)}

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clip_horizontal(fh, mu, fzL, fzR):
        """Clip horizontal forces to linearised friction cone."""
        maxL = mu * fzL
        maxR = mu * fzR
        fh[0] = np.clip(fh[0], -maxL, maxL)
        fh[1] = np.clip(fh[1], -maxL, maxL)
        fh[2] = np.clip(fh[2], -maxR, maxR)
        fh[3] = np.clip(fh[3], -maxR, maxR)
        return fh

    @staticmethod
    def _compute_mu_utilized(fh, mu, fzL, fzR):
        """Compute max friction utilisation across both legs."""
        util_L = np.sqrt(fh[0]**2 + fh[1]**2) / (mu * fzL) if mu * fzL > 0 else 0.0
        util_R = np.sqrt(fh[2]**2 + fh[3]**2) / (mu * fzR) if mu * fzR > 0 else 0.0
        return max(util_L, util_R)

    def _scale_wrench_to_feasible(self, A, b, fh_current, mu, fzL, fzR):
        """When the desired wrench is infeasible, scale it down until feasible.

        Binary search for scale factor λ ∈ [0, 1] such that the minimum-norm
        solution to A·f = λ·b satisfies friction constraints.
        """
        lo, hi = 0.0, 1.0
        best = fh_current.copy()
        for _ in range(20):
            mid = (lo + hi) / 2.0
            f_p_scaled = np.linalg.pinv(A) @ (mid * b)
            f_clipped = self._clip_horizontal(f_p_scaled.copy(), mu, fzL, fzR)
            if np.allclose(f_p_scaled, f_clipped, atol=1e-10):
                best = f_clipped
                lo = mid  # feasible, try larger
            else:
                hi = mid  # infeasible, try smaller
        return best
