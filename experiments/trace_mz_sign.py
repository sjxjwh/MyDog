"""Systematic Mz sign convention trace through the full force chain.

Reports the sign at each layer for a single stance configuration,
allowing isolation of sign flips.
"""

import numpy as np

# ── Simulate a single stance with controlled inputs ──
# Use FR+RL trot diagonal stance at t=0
# FR at (0.19, -0.094) relative to COM, RL at (-0.19, 0.094)

r_FR = np.array([0.19, -0.094, 0.0])
r_RL = np.array([-0.19, 0.094, 0.0])

print("=" * 70)
print("  Systematic Mz Sign Convention Trace")
print("=" * 70)

# ──── Layer 1: PD output ────
# Simulate: target_wz=0.3, actual_wz=0.03, yaw_err=0.1 rad
target_wz = 0.3
actual_wz = 0.03
yaw_err = 0.1  # small angle error
Izz = 0.423  # corrected full-system

Kp_yaw = 90.0  # rate PD
Kp_angle = 30.0  # angle P
Ki_angle = 12.0  # angle I
yaw_int = 0.05  # small integral

Mz_raw = Kp_yaw*(target_wz - actual_wz) + Kp_angle*yaw_err + Ki_angle*yaw_int
print(f"\n  Layer 1 — PD Output:")
print(f"    Mz_raw = {Mz_raw:.2f} Nm  (+ = CCW body torque desired)")
print(f"    Want: body to yaw CCW (wz=0.03 → wz=0.3)")

# ──── Layer 2: Wrench Mz (before distributor) ────
# Option A: Mz_wrench = Mz_raw (as currently coded)
# Option B: Mz_wrench = -Mz_raw (body_torque → foot_torque)
Mz_wrench_A = Mz_raw   # current code
Mz_wrench_B = -Mz_raw  # proposed fix

print(f"\n  Layer 2 — Wrench Mz:")
print(f"    Option A (current): Mz_wrench = Mz_raw = {Mz_wrench_A:.2f} Nm")
print(f"    Option B (proposed): Mz_wrench = -Mz_raw = {Mz_wrench_B:.2f} Nm")

# ──── Layer 3: Force distributor solves A·f = [0, 0, Mz] ────
# For pure Mz (no Fx, Fy), the minimum-norm solution is:
# f = A_pinv @ [0, 0, Mz]
# where A = [[1,0,1,0], [0,1,0,1], [-rFR_y, rFR_x, -rRL_y, rRL_x]]

A = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [-r_FR[1], r_FR[0], -r_RL[1], r_RL[0]],
])

for label, Mz_w in [("A (current)", Mz_wrench_A), ("B (proposed)", Mz_wrench_B)]:
    b = np.array([0.0, 0.0, Mz_w])
    A_pinv = np.linalg.pinv(A)
    f = A_pinv @ b
    fx_FR, fy_FR, fx_RL, fy_RL = f

    # Computed Mz from these forces: r×f
    Mz_from_f = (-r_FR[1]*fx_FR + r_FR[0]*fy_FR
                 - r_RL[1]*fx_RL + r_RL[0]*fy_RL)

    print(f"\n  Layer 3 — Distributor Solution ({label}):")
    print(f"    Mz_wrench = {Mz_w:.2f} Nm")
    print(f"    FR: fx={fx_FR:+.2f}, fy={fy_FR:+.2f}  (foot→ground)")
    print(f"    RL: fx={fx_RL:+.2f}, fy={fy_RL:+.2f}  (foot→ground)")
    print(f"    Verified Mz from solution = {Mz_from_f:.2f} Nm ✓")

    # ──── Layer 4: foot→ground forces → impedance ────
    # The foot forces go through apply_impedance: τ = Jᵀ @ (Kp*err - Kd*v + ff)
    # In steady state (no displacement, no velocity): τ = Jᵀ @ ff
    # The joint torques act on the BODY through the kinematic chain.
    # For a simplified analysis: body torque ≈ -(r × ff)
    # (the foot pushes the ground, the ground pushes back on the body)

    body_torque = -(r_FR[0]*fy_FR - r_FR[1]*fx_FR
                    + r_RL[0]*fy_RL - r_RL[1]*fx_RL)
    # Note: this is -(r×f) because f is foot→ground, body gets reaction

    print(f"\n  Layer 4 — Body Torque from these forces:")
    print(f"    body_torque = -(r×f) = {body_torque:.2f} Nm")
    print(f"    (positive = CCW)")

    if body_torque > 0.1:
        print(f"    → Body would yaw CCW (toward desired direction) ✓" if target_wz > actual_wz else "    → Body would yaw CCW")
    elif body_torque < -0.1:
        print(f"    → Body would yaw CW (OPPOSITE to desired) ✗" if target_wz > actual_wz else "    → Body would yaw CW")

    # ──── Layer 5: What happens in impedance? ────
    # In apply_impedance: F = Kp*(target - x) - Kd*v + ff
    # Stance target = body_pos + R_body @ offset_body
    # If body doesn't move, target ≈ foot_pos → Kp term ≈ 0
    # If foot tries to move due to ff, Kp term opposes

    print(f"\n  Layer 5 — Impedance Effect:")
    stance_Kp_x = 100.0  # current stance Kp in x
    # To achieve fx=10N, foot would need displacement dx = fx/Kp ≈ 10/100 = 0.1m
    # But foot is planted → displacement constrained → net force < ff
    print(f"    Stance Kp_x = {stance_Kp_x} N/m")
    print(f"    To sustain fx={abs(fx_FR):.1f}N: dx = {abs(fx_FR)/stance_Kp_x*100:.1f} cm displacement needed")
    print(f"    → Impedance Kp term OPPOSES feed-forward → net force ≪ ff")
    print(f"    → This is the root cause of foot-force yaw inefficiency")

print(f"\n{'='*70}")
print("  CONCLUSION")
print(f"{'='*70}")
print(f"  Option A (current): Mz_wrench = Mz_raw")
print(f"    → If correct sign through all layers: body_torque should ≈ -Mz_raw")
print(f"    → Body receives OPPOSITE torque to what PD commands")
print(f"    → Integrator winds up, wz tracking fails")
print(f"")
print(f"  Option B (proposed): Mz_wrench = -Mz_raw")
print(f"    → Body torque should ≈ Mz_raw (correct direction)")
print(f"    → But impedance Kp term still opposes ff")
print(f"    → Foot-force yaw still inefficient, but at least correct direction")
print(f"")
print(f"  Fix: either negate Mz in wrench AND reduce stance Kp to ~10 N/m,")
print(f"       or use direct yaw (qwfrc_applied) which bypasses both issues.")
print(f"{'='*70}")
