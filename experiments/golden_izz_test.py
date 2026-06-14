"""Golden experiment v2: compute effective yaw inertia from full mass matrix.

Uses MuJoCo's mj_fullM to get the joint-space mass matrix M(q).
The effective inertia for yaw rotation is M[dof_adr+5, dof_adr+5]
which includes ALL body contributions (trunk + 4 legs) via parallel axis theorem.
"""

import numpy as np
import mujoco

MODEL_PATH = "/home/scj/MyDog/model/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ── Set legs to default standing configuration ──
HOME = np.array([0.0, 0.9, -1.8])
ALL_LEGS = ["FR", "FL", "RR", "RL"]
for leg in ALL_LEGS:
    for j, joint in enumerate(["hip", "thigh", "calf"]):
        jname = f"{leg}_{joint}_joint"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qpos_adr = model.jnt_qposadr[jid]
            if qpos_adr >= 0 and qpos_adr < model.nq:
                data.qpos[qpos_adr] = HOME[j]

mujoco.mj_forward(model, data)

print("=" * 70)
print("  Golden Experiment v2: Full Mass Matrix Yaw Inertia")
print("=" * 70)

# ── Compute full mass matrix ──
# mj_fullM returns the mass matrix in qM (nv × nv)
nv = model.nv
qM = np.zeros((nv, nv))
mujoco.mj_fullM(model, qM, data.qM)

# Freejoint DOFs
trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
dof_adr = model.jnt_dofadr[model.body_jntadr[trunk_id]]

print(f"\n  Full mass matrix shape: {nv}×{nv}")
print(f"  Freejoint DOFs: dof_adr={dof_adr}..{dof_adr+5}")
print(f"  Leg DOFs: {dof_adr+6}..{nv-1}")

# ── Extract effective yaw inertia ──
# M[dof_adr+5, dof_adr+5] is the inertia felt at the yaw DOF
# This INCLUDES all coupling through the kinematic chain
M_yaw_yaw = qM[dof_adr + 5, dof_adr + 5]
M_yaw_legs = qM[dof_adr + 5, dof_adr + 6:nv]
M_legs_legs = qM[dof_adr + 6:nv, dof_adr + 6:nv]

print(f"\n  ── Yaw Inertia Components ──")
print(f"  M[wz, wz] (effective yaw inertia) = {M_yaw_yaw:.6f} kg·m²")
print(f"  Cross-coupling norm ||M[wz, legs]|| = {np.linalg.norm(M_yaw_legs):.4f}")
print(f"  Leg submatrix norm ||M[legs, legs]|| = {np.linalg.norm(M_legs_legs):.4f}")

# ── Lock legs and recompute ──
# If we lock the legs (remove leg DOFs), the effective inertia reduces
# to the trunk-only value. Let's compute the Schur complement to see
# what inertia the trunk feels when legs are free to move:
# Ieff = M[5,5] - M[5,legs] @ inv(M[legs,legs]) @ M[legs,5]
try:
    M_legs_inv = np.linalg.inv(M_legs_legs)
    schur = M_yaw_yaw - M_yaw_legs @ M_legs_inv @ M_yaw_legs.T
    print(f"\n  ── Leg-decoupled yaw inertia (Schur complement) ──")
    print(f"  Ieff (legs free) = {schur:.6f} kg·m²")
    print(f"  This is what the trunk 'feels' when legs can swing freely.")
except np.linalg.LinAlgError:
    schur = None
    print("  Schur complement: singular (leg submatrix not invertible)")

# ── Compare with controller value ──
TRUNK_I = np.array([0.07166, 0.06301, 0.01681])
print(f"\n  ── Comparison ──")
print(f"  Controller TRUNK_I[2] (used in MPC)  = {TRUNK_I[2]:.6f} kg·m²")
print(f"  MuJoCo body_inertia[trunk][2]        = {model.body_inertia[trunk_id][2]:.6f} kg·m²")
print(f"  Mass matrix M[wz,wz] (full system)   = {M_yaw_yaw:.6f} kg·m²")
if schur is not None:
    print(f"  Schur complement (legs decoupled)    = {schur:.6f} kg·m²")

ratio = M_yaw_yaw / TRUNK_I[2]
print(f"\n  ── RATIO ──")
print(f"  Full-system Izz / controller Izz = {ratio:.2f}x")
print(f"  Controller uses ONLY trunk inertia, missing leg contributions.")

# ── Print per-body contributions ──
print(f"\n  ── Per-body mass properties ──")
print(f"  {'Body':<20} {'mass (kg)':<12} {'Ixx':<12} {'Iyy':<12} {'Izz':<12}")
print(f"  {'-'*68}")
total_m = 0
for i in range(model.nbody):
    m = model.body_mass[i]
    if m > 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        I = model.body_inertia[i]
        total_m += m
        # Body position relative to trunk COM
        r = data.xipos[i] - data.xipos[trunk_id]
        # Parallel axis contribution to Izz about trunk COM
        Izz_pa = m * (r[0]**2 + r[1]**2)  # parallel axis: Izz += m * d_xy^2
        print(f"  {name:<20} {m:<12.4f} {I[0]:<12.6f} {I[1]:<12.6f} {I[2]:<12.6f} "
              f"d_xy={np.sqrt(r[0]**2+r[1]**2):.3f}m, PA_contrib={Izz_pa:.4f}")

# ── Conclusion ──
print(f"\n{'='*70}")
print("  CONCLUSION")
print(f"{'='*70}")
if ratio > 1.5:
    print(f"  🔴 The controller uses Izz = {TRUNK_I[2]:.6f} (TRUNK ONLY).")
    print(f"     The actual full-system yaw inertia is M[wz,wz] = {M_yaw_yaw:.3f}.")
    print(f"     Ratio = {ratio:.1f}x — the controller underestimates yaw inertia.")
    print(f"")
    print(f"  This explains the Contact→Inertial gap:")
    print(f"  - Mz_contact is computed from r×f in world frame (correct)")
    print(f"  - Mz_inertial = Izz_controller × αz uses wrong Izz")
    print(f"  - Gap = Mz_contact × (1 − Izz_controller / Izz_true)")
    print(f"        ≈ Mz_contact × (1 − 1/{ratio:.0f})")
    print(f"        ≈ Mz_contact × {1 - 1/ratio:.1%}")
    print(f"")
    print(f"  FIX: Replace TRUNK_I[2] = 0.01681 with full-system Izz = {M_yaw_yaw:.4f}")
    print(f"  (or the Schur complement = {schur:.4f} for leg-decoupled dynamics)")
else:
    print(f"  Izz value is approximately correct.")
print(f"{'='*70}")
