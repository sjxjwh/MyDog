#!/usr/bin/env python3
"""Main entry point for single-foot trajectory simulation.

Loads a robot model (MJCF or URDF), generates a Cartesian trajectory for
one foot, and simulates trajectory tracking via IK-based position control.

Two modes:
  - Viewer mode (--viewer):   Interactive 3D GUI with ground, sky, lighting
  - Headless mode (default):  Batch simulation + 4 analysis plots in output/

Usage:
    python -m src.main --viewer                  # Interactive 3D viewer
    python -m src.main                           # Headless: FR leg, circle traj
    python -m src.main --leg FR --traj circle    # Circle trajectory
    python -m src.main --leg FR --traj line      # Linear trajectory
    python -m src.main --urdf --leg FL --traj circle  # URDF input
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .simulator import MuJoCoSim, create_fixed_base_mjcf
from .controller import IKFootController, create_go1_leg_kinematics
from .trajectory import (
    CircleTrajectory,
    LinearTrajectory,
    SinusoidalTrajectory,
    LissajousTrajectory,
)
from .urdf_loader import urdf_to_mjcf_xml
from .gait import GaitType, GaitParams
from .body_controller import BodyController
from .force_controller import MITBodyController, GO1_MASS

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Default model paths
MJCF_PATH = ROOT / "model" / "go1.xml"
MJCF_FIXED_PATH = ROOT / "model" / "go1_fixed.xml"
SCENE_FIXED_PATH = ROOT / "model" / "scene_fixed.xml"
SCENE_PATH = ROOT / "model" / "scene.xml"          # floating base + ground
URDF_PATH = ROOT / "model" / "unitree" / "go1.urdf"


def prepare_model(use_urdf: bool = False, with_scene: bool = False) -> str:
    """Prepare the MuJoCo MJCF model for simulation.

    If use_urdf is True, converts the Unitree URDF to MJCF first.
    Otherwise, creates a fixed-base version of the existing go1.xml.

    Args:
        use_urdf: If True, convert URDF to MJCF.
        with_scene: If True, return the scene XML path (with ground, sky,
                    lighting) instead of the bare model. Only applies when
                    not using URDF and not using --model.

    Returns:
        Path to the MJCF XML file to load.
    """
    if use_urdf:
        out_path = str(ROOT / "model" / "go1_from_urdf.xml")
        print(f"Converting URDF to MJCF: {URDF_PATH} → {out_path}")
        urdf_to_mjcf_xml(str(URDF_PATH), out_path, base_link="trunk", fix_base=True)
        return out_path

    # Create fixed-base variant if it doesn't exist
    if not MJCF_FIXED_PATH.exists():
        print(f"Creating fixed-base MJCF: {MJCF_PATH} → {MJCF_FIXED_PATH}")
        create_fixed_base_mjcf(str(MJCF_PATH), str(MJCF_FIXED_PATH),
                               base_body="trunk", base_height=0.445)
    else:
        print(f"Using existing fixed-base model: {MJCF_FIXED_PATH}")

    if with_scene:
        if not SCENE_FIXED_PATH.exists():
            print(f"Scene file not found: {SCENE_FIXED_PATH}")
            print("Please ensure model/scene_fixed.xml exists in the project.")
        return str(SCENE_FIXED_PATH)

    return str(MJCF_FIXED_PATH)


def generate_trajectory(traj_type: str, home_pos: np.ndarray, dt: float):
    """Generate a foot trajectory.

    Args:
        traj_type: Type of trajectory ('circle', 'line', 'sine', 'lissajous').
        home_pos: The resting position of the foot (used as trajectory center).
        dt: Time step for sampling.

    Returns:
        Tuple of (sampled_points, trajectory_object, duration).
    """
    T = 2.0  # Duration in seconds

    if traj_type == "circle":
        traj = CircleTrajectory(
            center=home_pos + np.array([0.0, 0.0, 0.02]),
            radius=0.04,
            axis="z",
            T=T,
        )
    elif traj_type == "line":
        traj = LinearTrajectory(
            start=home_pos,
            end=home_pos + np.array([0.04, 0.03, 0.04]),
            T=T,
        )
    elif traj_type == "sine":
        traj = SinusoidalTrajectory(
            center=home_pos,
            amplitude=np.array([0.02, 0.02, 0.03]),
            frequency=1.0,
            T=T,
        )
    elif traj_type == "lissajous":
        traj = LissajousTrajectory(
            center=home_pos,
            amplitude=np.array([0.03, 0.02, 0.02]),
            frequencies=(2, 3, 1),
            T=T,
        )
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    points = traj.sample(dt)
    return points, traj, T


def run_simulation(model_path: str, leg: str, traj_type: str,
                   dt: float = 0.002, record: bool = True):
    """Run the trajectory tracking simulation.

    Args:
        model_path: Path to MJCF XML file.
        leg: Leg identifier ('FR', 'FL', 'RR', 'RL').
        traj_type: Type of trajectory.
        dt: Simulation time step.
        record: Whether to record data for plotting.

    Returns:
        Recorded data dict if record=True, else None.
    """
    print(f"\n{'='*60}")
    print(f"  MyDog — Single Foot Trajectory Simulation")
    print(f"  Leg: {leg}, Trajectory: {traj_type}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    sim = MuJoCoSim(model_path)
    print(f"Model loaded: {sim._model.nq} qpos, {sim._model.nv} qvel, {sim._model.nu} actuators")

    # List available actuators
    actuator_names = sim.actuator_names()
    print(f"Available actuators: {actuator_names}")

    # Create kinematics and controller
    kin = create_go1_leg_kinematics(leg)
    controller = IKFootController(sim, leg, kin)

    leg_joints = controller.joint_names
    print(f"Leg joints: {leg_joints}")

    # Set initial joint positions to home configuration
    # Go1 home: abduction=0, hip=0.9, knee=-1.8 for each leg
    home_angles = [0.0, 0.9, -1.8]
    for jname, angle in zip(leg_joints, home_angles):
        sim.set_qpos(jname, angle)

    # Forward kinematics to get home foot position
    sim.forward()
    home_foot = controller.get_foot_position()
    print(f"Home foot position (world): [{home_foot[0]:.4f}, {home_foot[1]:.4f}, {home_foot[2]:.4f}]")

    # Compute MuJoCo-vs-analytical FK offset (critical for new analytical IK).
    # The MJCF model has a thigh lateral offset [0, ±0.08, 0] that the
    # analytical LegKinematics ignores — it is constant in hip frame when the
    # abduction angle is near zero (typical for locomotion).
    hip_pos = controller.get_hip_frame_position()
    hip_rot = sim.get_body_rotation(f"{leg}_hip")
    foot_hip = hip_rot.T @ (home_foot - hip_pos)
    foot_analytic_hip, _ = kin.forward_kinematics(home_angles)
    controller.set_mujoco_offset(foot_hip - foot_analytic_hip)

    # Generate trajectory
    traj_points, traj_obj, T = generate_trajectory(traj_type, home_foot, dt)
    n_steps = len(traj_points)
    n_cycles = 3  # Repeat trajectory N times
    total_steps = n_steps * n_cycles

    print(f"Trajectory: {n_steps} waypoints over {T:.1f}s, repeating {n_cycles}x "
          f"(total: {total_steps} steps)")

    # Recording arrays
    if record:
        data = {
            "time": np.zeros(total_steps),
            "target": np.zeros((total_steps, 3)),
            "actual": np.zeros((total_steps, 3)),
            "error": np.zeros((total_steps, 3)),
            "joint_targets": np.zeros((total_steps, 3)),
            "joint_actual": np.zeros((total_steps, 3)),
        }

    # Simulation loop
    print("\nRunning simulation...")
    ik_failures = 0

    for cycle in range(n_cycles):
        for i in range(n_steps):
            idx = cycle * n_steps + i
            target_world = traj_points[i]

            # Solve IK and set control
            q_target = controller.control(target_world)
            if q_target is None:
                ik_failures += 1
                # Hold last position
                pass

            # Step simulation
            sim.step()

            # Get actual foot position
            actual_world = controller.get_foot_position()

            if record:
                data["time"][idx] = idx * dt
                data["target"][idx] = target_world
                data["actual"][idx] = actual_world
                data["error"][idx] = actual_world - target_world
                if q_target is not None:
                    data["joint_targets"][idx] = q_target
                data["joint_actual"][idx] = controller.get_current_joint_angles()

    if ik_failures > 0:
        print(f"Warning: {ik_failures}/{total_steps} IK solutions failed")
    else:
        print("All IK solutions converged successfully.")

    # Final error stats
    if record:
        rmse = np.sqrt(np.mean(data["error"] ** 2, axis=0))
        max_err = np.max(np.abs(data["error"]), axis=0)
        print(f"\nTracking Error (xyz):")
        print(f"  RMSE:     [{rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f}] m")
        print(f"  Max Abs:  [{max_err[0]:.4f}, {max_err[1]:.4f}, {max_err[2]:.4f}] m")

    print("Simulation complete.\n")
    return data if record else None


def run_simulation_viewer(model_path: str, leg: str, traj_type: str,
                         dt: float = 0.002):
    """Run trajectory tracking with the MuJoCo interactive viewer.

    Opens a real-time 3D GUI window showing the robot and foot trajectory.
    Press Space to pause/resume, scroll to zoom, right-drag to rotate.

    Args:
        model_path: Path to MJCF XML file.
        leg: Leg identifier ('FR', 'FL', 'RR', 'RL').
        traj_type: Type of trajectory.
        dt: Simulation time step.
    """
    import mujoco.viewer

    print(f"\n{'='*60}")
    print(f"  MyDog — Interactive Viewer Mode")
    print(f"  Leg: {leg}, Trajectory: {traj_type}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")
    print("Controls: Space=pause, Scroll=zoom, Right-drag=rotate, "
          "Ctrl+Right-drag=pan\n")

    # Load model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create sim wrapper (lightweight, no reload)
    sim = MuJoCoSim.__new__(MuJoCoSim)
    sim._model = model
    sim._data = data
    sim.model_path = Path(model_path)
    sim._body_ids = {}
    sim._joint_ids = {}
    sim._actuator_ids = {}
    sim._site_ids = {}
    sim._build_name_index()

    # Create kinematics and controller
    kin = create_go1_leg_kinematics(leg)
    controller = IKFootController(sim, leg, kin)

    leg_joints = controller.joint_names
    print(f"Leg joints: {leg_joints}")

    # Set initial joint positions
    home_angles = [0.0, 0.9, -1.8]
    for jname, angle in zip(leg_joints, home_angles):
        sim.set_qpos(jname, angle)

    sim.forward()
    home_foot = controller.get_foot_position()
    print(f"Home foot position: [{home_foot[0]:.4f}, {home_foot[1]:.4f}, {home_foot[2]:.4f}]")

    # Compute MuJoCo-vs-analytical FK offset (same as run_simulation above)
    hip_pos = controller.get_hip_frame_position()
    hip_rot = sim.get_body_rotation(f"{leg}_hip")
    foot_hip = hip_rot.T @ (home_foot - hip_pos)
    foot_analytic_hip, _ = kin.forward_kinematics(home_angles)
    controller.set_mujoco_offset(foot_hip - foot_analytic_hip)

    # Generate trajectory
    traj_points, traj_obj, T = generate_trajectory(traj_type, home_foot, dt)
    n_steps = len(traj_points)

    # Launch passive viewer
    ik_failures = 0
    cycle = 0
    i = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Get current target
            target_world = traj_points[i]

            # Solve IK and apply control
            q_target = controller.control(target_world)
            if q_target is None:
                ik_failures += 1

            # Step physics
            mujoco.mj_step(model, data)

            # Sync viewer (renders at display refresh rate)
            viewer.sync()

            # Advance trajectory index
            i += 1
            if i >= n_steps:
                i = 0
                cycle += 1
                if cycle % 10 == 0:
                    print(f"  Cycle {cycle}...")

    if ik_failures > 0:
        print(f"Warning: {ik_failures} IK failures occurred")
    print("Viewer closed.\n")


def plot_results(data: dict, leg: str, traj_type: str, output_dir: Path):
    """Plot the simulation results."""
    print(f"Plotting results to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    time = data["time"]
    target = data["target"]
    actual = data["actual"]
    error = data["error"]

    # 1. 3D trajectory comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], "b-", linewidth=1.5, label="Target")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "r--", linewidth=1.0, label="Actual")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{leg} Foot Trajectory Tracking ({traj_type})")
    ax.legend()
    fig.savefig(output_dir / f"trajectory_3d_{leg}_{traj_type}.png", dpi=150)
    plt.close(fig)

    # 2. Position vs Time
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for j, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, target[:, j], "b-", linewidth=1.0, label="Target")
        ax.plot(time, actual[:, j], "r--", linewidth=0.8, label="Actual")
        ax.set_ylabel(f"{label} (m)")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{leg} Foot Position vs Time ({traj_type})")
    fig.tight_layout()
    fig.savefig(output_dir / f"position_time_{leg}_{traj_type}.png", dpi=150)
    plt.close(fig)

    # 3. Tracking Error vs Time
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for j, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, error[:, j] * 1000, "r-", linewidth=0.5)  # mm
        ax.set_ylabel(f"{label} Error (mm)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{leg} Foot Tracking Error ({traj_type})")
    fig.tight_layout()
    fig.savefig(output_dir / f"error_time_{leg}_{traj_type}.png", dpi=150)
    plt.close(fig)

    # 4. Joint Angles
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    jlabels = ["Abduction", "Hip Pitch", "Knee Pitch"]
    for j, (ax, label) in enumerate(zip(axes, jlabels)):
        ax.plot(time, data["joint_targets"][:, j], "b-", linewidth=1.0, label="Target")
        ax.plot(time, data["joint_actual"][:, j], "r--", linewidth=0.8, label="Actual")
        ax.set_ylabel(f"{label} (rad)")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{leg} Joint Angles ({traj_type})")
    fig.tight_layout()
    fig.savefig(output_dir / f"joints_{leg}_{traj_type}.png", dpi=150)
    plt.close(fig)

    print(f"  ✓ trajectory_3d_{leg}_{traj_type}.png")
    print(f"  ✓ position_time_{leg}_{traj_type}.png")
    print(f"  ✓ error_time_{leg}_{traj_type}.png")
    print(f"  ✓ joints_{leg}_{traj_type}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Gait simulation functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_gait_simulation(model_path: str, gait_type: GaitType,
                        params: GaitParams, dt: float = 0.002,
                        total_cycles: float = 5.0, floating: bool = False,
                        warm_up: float = 0.2):
    """Run a full-body gait simulation (headless mode).

    Coordinates all four legs through a periodic gait pattern and records
    tracking data for analysis.

    Args:
        model_path: Path to MJCF XML file.
        gait_type: Type of gait (TROT, WALK, PACE, BOUND).
        params: Gait parameters (cycle period, duty factor, etc.).
        dt: Simulation time step.
        total_cycles: Number of gait cycles to simulate.
        floating: If True, use floating-base model (dog moves on ground).

    Returns:
        Recorded data dict.
    """
    print(f"\n{'='*60}")
    print(f"  MyDog — Full-Body Gait Simulation"
          f"{' [FLOATING]' if floating else ''}")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Step length={params.step_length:.3f}m, "
          f"step height={params.step_height:.3f}m")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    sim = MuJoCoSim(model_path)
    print(f"Model loaded: {sim.nq} qpos, {sim.nv} qvel, {sim.nu} actuators")
    print(f"Available actuators: {sim.actuator_names()}")

    # Create body controller
    bc = BodyController(sim, gait_type, params, floating=floating,
                        warm_up=warm_up)
    print(bc.summary())

    # Floating base: let the body settle on the ground first
    if floating:
        print("Settling body on ground (0.3s)...")
        bc.settle(duration=0.3, dt=dt)

    # Timing
    T_total = total_cycles * params.T_cycle
    total_steps = int(T_total / dt)
    print(f"\nSimulating {total_cycles:.1f} cycles ({T_total:.1f}s) "
          f"at dt={dt}s → {total_steps} steps")

    # Start recording
    bc.start_recording(total_steps)

    # Simulation loop
    print("Running simulation...")
    for step_idx in range(total_steps):
        t = step_idx * dt
        bc.control(t)
        bc.step()

    # Summary
    total_failures = sum(bc.ik_failures.values())
    if total_failures > 0:
        print(f"Warning: {total_failures}/{total_steps * 4} IK solutions failed "
              f"({bc.ik_failures})")
    else:
        print("All IK solutions converged successfully.")

    # Error stats
    data = bc.get_recorded_data()
    for leg in ["FR", "FL", "RR", "RL"]:
        error = data["actual"][leg] - data["target"][leg]
        rmse = np.sqrt(np.mean(error ** 2, axis=0))
        print(f"  {leg} RMSE: [{rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f}] m")

    print("Simulation complete.\n")
    return data


def run_gait_simulation_viewer(model_path: str, gait_type: GaitType,
                               params: GaitParams, dt: float = 0.002,
                               floating: bool = False):
    """Run full-body gait simulation with the MuJoCo interactive viewer.

    Opens a real-time 3D GUI window showing the robot walking with all four
    legs coordinated. Uses substeps to maintain real-time physics at the
    viewer's display refresh rate (~60 Hz).

    Controls: Space=pause, Scroll=zoom, Right-drag=rotate, Ctrl+Right-drag=pan

    Args:
        model_path: Path to MJCF XML file.
        gait_type: Type of gait.
        params: Gait parameters.
        dt: Simulation time step.
        floating: If True, use floating-base model (dog moves on ground).
    """
    import mujoco.viewer

    print(f"\n{'='*60}")
    print(f"  MyDog — Interactive Gait Viewer")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Step length={params.step_length:.3f}m, "
          f"step height={params.step_height:.3f}m")
    if floating:
        print(f"  Mode: FLOATING BASE (dog runs on ground)")
    print(f"{'='*60}\n")
    print("Controls: Space=pause, Scroll=zoom, Right-drag=rotate, "
          "Ctrl+Right-drag=pan\n")

    # Load model and data directly (lightweight wrapper)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create sim wrapper (lightweight, no reload)
    sim = MuJoCoSim.__new__(MuJoCoSim)
    sim._model = model
    sim._data = data
    sim.model_path = Path(model_path)
    sim._body_ids = {}
    sim._joint_ids = {}
    sim._actuator_ids = {}
    sim._site_ids = {}
    sim._build_name_index()

    # Create body controller
    bc = BodyController(sim, gait_type, params, floating=floating)
    print(bc.summary())

    # Floating base: settle body on ground before starting gait
    if floating:
        print("Settling body on ground (0.3s)...")
        bc.settle(duration=0.3, dt=dt)

    # Substep count: enough physics steps per viewer frame for real-time
    viewer_fps = 60.0
    substeps = max(1, int(1.0 / (dt * viewer_fps)))
    print(f"Viewer substeps: {substeps} (dt={dt}s, fps≈{viewer_fps})\n")

    # Launch passive viewer
    t = 0.0
    ik_failures = 0
    cycle_count = 0
    last_cycle = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(substeps):
                bc.control(t)
                mujoco.mj_step(model, data)
                t += dt

            viewer.sync()

            # Log cycle progress
            current_cycle = int(t / params.T_cycle)
            if current_cycle > last_cycle:
                last_cycle = current_cycle
                if current_cycle % 10 == 0:
                    total_failures = sum(bc.ik_failures.values())
                    if total_failures > 0:
                        print(f"  Cycle {current_cycle} (t={t:.1f}s) — "
                              f"IK failures: {total_failures}")
                    else:
                        print(f"  Cycle {current_cycle} (t={t:.1f}s)")

    total_failures = sum(bc.ik_failures.values())
    if total_failures > 0:
        print(f"Warning: {total_failures} IK failures occurred "
              f"({bc.ik_failures})")
    print("Viewer closed.\n")


def run_gait_force_viewer(model_path: str, gait_type: GaitType,
                          params: GaitParams, dt: float = 0.002,
                          target_vx: float = 0.3, target_vy: float = 0.0,
                          target_vyaw: float = 0.0):
    """Run force-controlled gait simulation with MuJoCo viewer.

    Uses MIT-style force control (MITBodyController) with body PD,
    force distribution, and Jacobian-transpose torque control.
    This is the proper dynamics simulation mode.

    Args:
        model_path: Path to MJCF XML file (must have freejoint + ground).
        gait_type: Type of gait.
        params: Gait parameters.
        dt: Simulation time step.
        target_vx: Desired forward velocity (m/s).
        target_vy: Desired lateral velocity (m/s).
        target_vyaw: Desired yaw rate (rad/s).
    """
    import mujoco.viewer

    print(f"\n{'='*60}")
    print(f"  MyDog — Force-Controlled Gait Viewer [MIT]")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Step length={params.step_length:.3f}m, "
          f"step height={params.step_height:.3f}m")
    print(f"  Target vx={target_vx:.2f}, vy={target_vy:.2f}, "
          f"vyaw={target_vyaw:.2f}")
    print(f"{'='*60}\n")
    print("Controls: Space=pause, Scroll=zoom, Right-drag=rotate, "
          "Ctrl+Right-drag=pan\n")

    # Load model and data directly
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create sim wrapper
    sim = MuJoCoSim.__new__(MuJoCoSim)
    sim._model = model
    sim._data = data
    sim.model_path = Path(model_path)
    sim._body_ids = {}
    sim._joint_ids = {}
    sim._actuator_ids = {}
    sim._site_ids = {}
    sim._build_name_index()

    # Create MIT force controller
    mc = MITBodyController(sim, gait_type, params)
    mc.target_vx = target_vx
    mc.target_vy = target_vy
    mc.target_vyaw = target_vyaw
    print(f"Body mass: {GO1_MASS:.1f} kg, weight: {GO1_MASS * 9.81:.0f} N")
    print(f"Controller ready.\n")

    # Settle body on ground using position control
    print("Settling body on ground (0.5s)...")
    mc.settle(duration=0.5, dt=dt)

    # Substep count
    viewer_fps = 60.0
    substeps = max(1, int(1.0 / (dt * viewer_fps)))
    print(f"Viewer substeps: {substeps} (dt={dt}s, fps≈{viewer_fps})\n")

    # Launch viewer
    t = 0.0
    cycle_count = 0
    last_cycle = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(substeps):
                mc.control(t)
                mujoco.mj_step(model, data)
                t += dt

            viewer.sync()

            # Log cycle progress
            current_cycle = int(t / params.T_cycle)
            if current_cycle > last_cycle:
                last_cycle = current_cycle
                if current_cycle % 5 == 0:
                    pos = data.qpos[0:3]
                    vel = data.qvel[0:3]
                    print(f"  Cycle {current_cycle} (t={t:.1f}s) — "
                          f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                          f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

    print("Viewer closed.\n")


def run_gait_force_headless(model_path: str, gait_type: GaitType,
                            params: GaitParams, dt: float = 0.002,
                            total_cycles: float = 5.0,
                            target_vx: float = 0.3, target_vy: float = 0.0,
                            target_vyaw: float = 0.0):
    """Run force-controlled gait simulation (headless mode).

    Uses MIT-style force control for proper dynamics.
    """
    print(f"\n{'='*60}")
    print(f"  MyDog — Force-Controlled Gait [MIT, headless]")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Target vx={target_vx:.2f}, vy={target_vy:.2f}, "
          f"vyaw={target_vyaw:.2f}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    sim = MuJoCoSim(model_path)
    print(f"Model loaded: {sim.nq} qpos, {sim.nv} qvel, {sim.nu} actuators")

    # Create MIT force controller
    mc = MITBodyController(sim, gait_type, params)
    mc.target_vx = target_vx
    mc.target_vy = target_vy
    mc.target_vyaw = target_vyaw
    print(f"Body mass: {GO1_MASS:.1f} kg, weight: {GO1_MASS * 9.81:.0f} N")

    # Settle
    print("Settling body on ground (0.5s)...")
    mc.settle(duration=0.5, dt=dt)

    # Run
    T_total = total_cycles * params.T_cycle
    total_steps = int(T_total / dt)
    print(f"\nSimulating {total_cycles:.1f} cycles ({T_total:.1f}s) "
          f"at dt={dt}s → {total_steps} steps")

    print("Running simulation...")
    for step_idx in range(total_steps):
        t = step_idx * dt
        mc.control(t)
        mc.step()

        if step_idx % 500 == 0:
            pos = sim._data.qpos[0:3]
            vel = sim._data.qvel[0:3]
            print(f"  Step {step_idx}/{total_steps}: "
                  f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                  f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

    # Print velocity tracking summary
    mc._metrics.print_summary()
    print("Simulation complete.\n")


def run_gait_mpc_viewer(model_path: str, gait_type: GaitType,
                         params: GaitParams, dt: float = 0.002,
                         target_vx: float = 0.3, target_vy: float = 0.0,
                         target_vyaw: float = 0.0):
    """Run MPC + MIT impedance-controlled gait simulation with MuJoCo viewer.

    Uses SRB convex MPC (SrbMpcSolver) to optimize ground reaction forces
    over a 0.3s horizon, feeding into MIT impedance control per leg.

    Args:
        model_path: Path to MJCF XML file (must have freejoint + ground).
        gait_type: Type of gait.
        params: Gait parameters.
        dt: Simulation time step.
        target_vx: Desired forward velocity (m/s).
        target_vy: Desired lateral velocity (m/s).
        target_vyaw: Desired yaw rate (rad/s).
    """
    import mujoco.viewer
    from .force_controller import MPCMITBodyController

    print(f"\n{'='*60}")
    print(f"  MyDog — MPC + MIT Impedance Gait Viewer")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Step length={params.step_length:.3f}m, "
          f"step height={params.step_height:.3f}m")
    print(f"  Target vx={target_vx:.2f}, vy={target_vy:.2f}, "
          f"vyaw={target_vyaw:.2f}")
    print(f"  MPC: 10-step horizon @ 30ms → 0.3s, OSQP solver")
    print(f"{'='*60}\n")
    print("Controls: Space=pause, Scroll=zoom, Right-drag=rotate, "
          "Ctrl+Right-drag=pan\n")

    # Load model and data directly
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create sim wrapper
    sim = MuJoCoSim.__new__(MuJoCoSim)
    sim._model = model
    sim._data = data
    sim.model_path = Path(model_path)
    sim._body_ids = {}
    sim._joint_ids = {}
    sim._actuator_ids = {}
    sim._site_ids = {}
    sim._build_name_index()

    # Create MPC + MIT controller
    mc = MPCMITBodyController(sim, gait_type, params)
    mc.target_vx = target_vx
    mc.target_vy = target_vy
    mc.target_vyaw = target_vyaw
    print(f"Body mass: {GO1_MASS:.1f} kg, weight: {GO1_MASS * 9.81:.0f} N")
    print(f"MPC horizon: 10 steps × 30ms = 0.3s")
    print(f"MPC freq: ~31 Hz (every {MPCMITBodyController.MPC_DECIMATION} "
          f"sim steps)\n")

    # Settle body on ground using position control
    print("Settling body on ground (0.5s)...")
    mc.settle(duration=0.5, dt=dt)

    # Substep count
    viewer_fps = 60.0
    substeps = max(1, int(1.0 / (dt * viewer_fps)))
    print(f"Viewer substeps: {substeps} (dt={dt}s, fps≈{viewer_fps})\n")

    # Launch viewer
    t = 0.0
    last_cycle = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(substeps):
                mc.control(t)
                mc.step()
                t += dt

            viewer.sync()

            # Log cycle progress
            current_cycle = int(t / params.T_cycle)
            if current_cycle > last_cycle:
                last_cycle = current_cycle
                if current_cycle % 5 == 0:
                    pos = data.qpos[0:3]
                    vel = data.qvel[0:3]
                    stats = mc.mpc_stats
                    print(f"  Cycle {current_cycle} (t={t:.1f}s) — "
                          f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                          f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] "
                          f"MPC: {stats['solve_time_ms']:.1f}ms"
                          f"{' [FB]' if stats['fallback_count'] > 0 else ''}")

    print("Viewer closed.\n")


def run_gait_mpc_headless(model_path: str, gait_type: GaitType,
                           params: GaitParams, dt: float = 0.002,
                           total_cycles: float = 5.0,
                           target_vx: float = 0.3, target_vy: float = 0.0,
                           target_vyaw: float = 0.0):
    """Run MPC + MIT impedance gait simulation (headless mode)."""
    from .force_controller import MPCMITBodyController

    print(f"\n{'='*60}")
    print(f"  MyDog — MPC + MIT Impedance Gait [headless]")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Target vx={target_vx:.2f}, vy={target_vy:.2f}, "
          f"vyaw={target_vyaw:.2f}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    sim = MuJoCoSim(model_path)
    print(f"Model loaded: {sim.nq} qpos, {sim.nv} qvel, {sim.nu} actuators")

    # Create MPC + MIT controller
    mc = MPCMITBodyController(sim, gait_type, params)
    mc.target_vx = target_vx
    mc.target_vy = target_vy
    mc.target_vyaw = target_vyaw
    print(f"Body mass: {GO1_MASS:.1f} kg, weight: {GO1_MASS * 9.81:.0f} N")

    # Settle
    print("Settling body on ground (0.5s)...")
    mc.settle(duration=0.5, dt=dt)

    # Run
    T_total = total_cycles * params.T_cycle
    total_steps = int(T_total / dt)
    print(f"\nSimulating {total_cycles:.1f} cycles ({T_total:.1f}s) "
          f"at dt={dt}s → {total_steps} steps")

    print("Running simulation...")
    for step_idx in range(total_steps):
        t = step_idx * dt
        mc.control(t)
        mc.step()

        if step_idx % 500 == 0:
            pos = sim._data.qpos[0:3]
            vel = sim._data.qvel[0:3]
            stats = mc.mpc_stats
            print(f"  Step {step_idx}/{total_steps}: "
                  f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                  f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] "
                  f"MPC: {stats['solve_time_ms']:.1f}ms "
                  f"FB: {stats['fallback_count']}/{stats['total_count']}")

    # Final stats
    stats = mc.mpc_stats
    print(f"\nMPC stats: avg solve {stats['solve_time_ms']:.1f}ms, "
          f"fallbacks {stats['fallback_count']}/{stats['total_count']} "
          f"({stats['fallback_rate']:.1%})")
    mc._metrics.print_summary()
    print("Simulation complete.\n")


def run_gait_quintic_viewer(model_path: str, gait_type: GaitType,
                             params: GaitParams, dt: float = 0.002,
                             target_vx: float = 0.3, target_vy: float = 0.0,
                             target_vyaw: float = 0.0,
                             mu_max: float = 0.6, adapt_params: bool = False,
                             use_momentum: bool = False):
    """Run quintic + friction/momentum-controlled gait simulation with MuJoCo viewer.

    Uses QuinticFrictionController or MomentumController.
    """
    import mujoco.viewer
    from .force_controller import QuinticFrictionController, MomentumController

    ctrl_name = "Momentum" if use_momentum else "Quintic + Friction"
    print(f"\n{'='*60}")
    print(f"  MyDog — {ctrl_name} Gait Viewer")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Step length={params.step_length:.3f}m, "
          f"step height={params.step_height:.3f}m")
    print(f"  Target vx={target_vx:.2f}, vy={target_vy:.2f}, "
          f"vyaw={target_vyaw:.2f}")
    print(f"  μ_max={mu_max}, adapt_params={adapt_params}")
    print(f"{'='*60}\n")
    print("Controls: Space=pause, Scroll=zoom, Right-drag=rotate, "
          "Ctrl+Right-drag=pan\n")

    # Load model and data directly
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create sim wrapper
    sim = MuJoCoSim.__new__(MuJoCoSim)
    sim._model = model
    sim._data = data
    sim.model_path = Path(model_path)
    sim._body_ids = {}
    sim._joint_ids = {}
    sim._actuator_ids = {}
    sim._site_ids = {}
    sim._build_name_index()

    # Create controller
    cls = MomentumController if use_momentum else QuinticFrictionController
    mc = cls(sim, gait_type, params, mu_max=mu_max, adapt_params=adapt_params)
    mc.target_vx = target_vx
    mc.target_vy = target_vy
    mc.target_vyaw = target_vyaw
    print(f"Body mass: {GO1_MASS:.1f} kg, weight: {GO1_MASS * 9.81:.0f} N")
    print(f"Controller ready.\n")

    # Settle body on ground using position control
    print("Settling body on ground (0.5s)...")
    mc.settle(duration=0.5, dt=dt)

    # Substep count
    viewer_fps = 60.0
    substeps = max(1, int(1.0 / (dt * viewer_fps)))
    print(f"Viewer substeps: {substeps} (dt={dt}s, fps≈{viewer_fps})\n")

    # Launch viewer
    t = 0.0
    cycle_count = 0
    last_cycle = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(substeps):
                mc.control(t)
                mujoco.mj_step(model, data)
                t += dt

            viewer.sync()

            # Log cycle progress with stats
            current_cycle = int(t / params.T_cycle)
            if current_cycle > last_cycle:
                last_cycle = current_cycle
                if current_cycle % 5 == 0:
                    pos = data.qpos[0:3]
                    vel = data.qvel[0:3]
                    fstats = mc.friction_stats
                    if 'mu_utilized' in fstats:
                        print(f"  Cycle {current_cycle} (t={t:.1f}s) — "
                              f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                              f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] "
                              f"μ={fstats['mu_utilized']:.2f} "
                              f"α={fstats['nullspace_alpha']:.1f} "
                              f"{'✓' if fstats['feasible'] else '⚠'}")
                    else:
                        cond_str = f"cond={fstats['cond_A']:.0f}" \
                            if fstats.get('cond_A') is not None else ""
                        print(f"  Cycle {current_cycle} (t={t:.1f}s) — "
                              f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                              f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] "
                              f"μ={fstats['mu_max_used']:.2f} {cond_str} "
                              f"{'✓' if fstats['feasible'] else '⚠'}")

    print("Viewer closed.\n")

    # ── Generate detailed debug report for yaw analysis ──
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    _report_dir2 = _Path(__file__).resolve().parent.parent / "output"
    _report_dir2.mkdir(parents=True, exist_ok=True)
    _ts2 = _dt.now().strftime("%Y%m%d_%H%M%S")
    _report_path2 = _report_dir2 / f"trot_yaw_debug_{_ts2}.md"
    mc.generate_debug_report(output_path=str(_report_path2))


def run_gait_quintic_headless(model_path: str, gait_type: GaitType,
                               params: GaitParams, dt: float = 0.002,
                               total_cycles: float = 5.0,
                               target_vx: float = 0.3, target_vy: float = 0.0,
                               target_vyaw: float = 0.0,
                               mu_max: float = 0.6, adapt_params: bool = False,
                               use_momentum: bool = False):
    """Run quintic + friction/momentum gait simulation (headless mode)."""
    from .force_controller import QuinticFrictionController, MomentumController

    ctrl_name = "Momentum" if use_momentum else "Quintic + Friction"
    print(f"\n{'='*60}")
    print(f"  MyDog — {ctrl_name} Gait [headless]")
    print(f"  Gait: {gait_type.value}, T_cycle={params.T_cycle:.2f}s, "
          f"duty={params.duty_factor:.2f}")
    print(f"  Target vx={target_vx:.2f}, vy={target_vy:.2f}, "
          f"vyaw={target_vyaw:.2f}")
    print(f"  μ_max={mu_max}, adapt_params={adapt_params}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # Load model
    sim = MuJoCoSim(model_path)
    print(f"Model loaded: {sim.nq} qpos, {sim.nv} qvel, {sim.nu} actuators")

    # Create controller
    cls = MomentumController if use_momentum else QuinticFrictionController
    mc = cls(sim, gait_type, params, mu_max=mu_max, adapt_params=adapt_params)
    mc.target_vx = target_vx
    mc.target_vy = target_vy
    mc.target_vyaw = target_vyaw
    print(f"Body mass: {GO1_MASS:.1f} kg, weight: {GO1_MASS * 9.81:.0f} N")

    # Settle
    print("Settling body on ground (0.5s)...")
    mc.settle(duration=0.5, dt=dt)

    # Run
    T_total = total_cycles * params.T_cycle
    total_steps = int(T_total / dt)
    print(f"\nSimulating {total_cycles:.1f} cycles ({T_total:.1f}s) "
          f"at dt={dt}s → {total_steps} steps")

    print("Running simulation...")
    for step_idx in range(total_steps):
        t = step_idx * dt
        mc.control(t)
        mc.step()

        if step_idx % 500 == 0:
            pos = sim._data.qpos[0:3]
            vel = sim._data.qvel[0:3]
            fstats = mc.friction_stats
            if 'mu_utilized' in fstats:
                print(f"  Step {step_idx}/{total_steps}: "
                      f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                      f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] "
                      f"μ={fstats['mu_utilized']:.2f} "
                      f"α={fstats['nullspace_alpha']:.1f} "
                      f"T={fstats['T_cycle']:.3f} L={fstats['step_length']:.3f}")
            else:
                cond_str = f"cond={fstats['cond_A']:.0f}" \
                    if fstats.get('cond_A') is not None else ""
                print(f"  Step {step_idx}/{total_steps}: "
                      f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
                      f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] "
                      f"μ={fstats['mu_max_used']:.2f} {cond_str} "
                      f"T={fstats['T_cycle']:.3f} L={fstats['step_length']:.3f}")

    # Final stats
    fstats = mc.friction_stats
    pos = sim._data.qpos[0:3]
    vel = sim._data.qvel[0:3]
    print(f"\nFinal state: "
          f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
          f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
    if 'mu_utilized' in fstats:
        print(f"Friction stats: μ_utilized={fstats['mu_utilized']:.3f}, "
              f"feasible={fstats['feasible']}, "
              f"α={fstats['nullspace_alpha']:.3f}")
    else:
        cond_str = f", cond_A={fstats['cond_A']:.1f}" \
            if fstats.get('cond_A') is not None else ""
        print(f"Momentum stats: μ_max={fstats['mu_max_used']:.3f}, "
              f"feasible={fstats['feasible']}{cond_str}")
    if adapt_params:
        print(f"Adapted params: T_cycle={fstats['T_cycle']:.3f}s, "
              f"step_length={fstats['step_length']:.3f}m")
    mc._metrics.print_summary()
    print("Simulation complete.\n")

    # ── Generate detailed debug report for yaw analysis ──
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    _report_dir = _Path(__file__).resolve().parent.parent / "output"
    _report_dir.mkdir(parents=True, exist_ok=True)
    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    _report_path = _report_dir / f"trot_yaw_debug_{_ts}.md"
    mc.generate_debug_report(output_path=str(_report_path))


def plot_gait_results(data: dict, gait_type: GaitType, output_dir: Path):
    """Plot gait simulation results."""
    print(f"Plotting gait results to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    time = data["time"]
    legs = ["FR", "FL", "RR", "RL"]
    gname = gait_type.value

    # ── 1. 3D foot trajectories (2×2 subplot per leg) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})
    for ax, leg in zip(axes.flat, legs):
        tgt = data["target"][leg]
        act = data["actual"][leg]
        ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], "b-", linewidth=0.8, alpha=0.7,
                label="Target")
        ax.plot(act[:, 0], act[:, 1], act[:, 2], "r--", linewidth=0.6,
                label="Actual")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"{leg} Foot")
        ax.legend(fontsize="x-small")
    fig.suptitle(f"Foot Trajectories — {gname} gait", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"gait_trajectory_3d_{gname}.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ gait_trajectory_3d_{gname}.png")

    # ── 2. Joint angles (4×1 subplot, 3 joints per leg) ──
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, leg in zip(axes, legs):
        jt = data["joint_targets"][leg]
        ja = data["joint_actual"][leg]
        for j, label in enumerate(["Abduction", "Hip", "Knee"]):
            ax.plot(time, jt[:, j], linewidth=0.8, alpha=0.6,
                    label=f"{label} target")
            ax.plot(time, ja[:, j], "--", linewidth=0.6,
                    label=f"{label} actual")
        ax.set_ylabel(f"{leg} (rad)")
        ax.legend(fontsize="xx-small", ncol=6, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Joint Angles — {gname} gait", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"gait_joints_{gname}.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ gait_joints_{gname}.png")

    # ── 3. Foot position components vs time (4 rows × 3 cols) ──
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey="row")
    for row, leg in enumerate(legs):
        tgt = data["target"][leg]
        act = data["actual"][leg]
        for col, (axis_name, axis_idx) in enumerate([("X", 0), ("Y", 1), ("Z", 2)]):
            ax = axes[row, col]
            ax.plot(time, tgt[:, axis_idx], "b-", linewidth=0.8, label="Target")
            ax.plot(time, act[:, axis_idx], "r--", linewidth=0.6, label="Actual")
            if row == 0:
                ax.set_title(f"{axis_name}")
            if col == 0:
                ax.set_ylabel(f"{leg}\n(m)")
            ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    fig.suptitle(f"Foot Position vs Time — {gname} gait", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"gait_position_time_{gname}.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ gait_position_time_{gname}.png")

    # ── 4. Gait phase diagram ──
    fig, ax = plt.subplots(figsize=(12, 4))
    leg_colors = {"FR": "#E74C3C", "FL": "#3498DB", "RR": "#2ECC71", "RL": "#F39C12"}
    bar_height = 0.8
    for i, leg in enumerate(legs):
        y = i * 1.0
        # Determine stance/swing transitions for phase visualization
        T_cycle = time[-1] / round(time[-1] / 0.5) if len(time) > 10 else 0.5
        T_cycle = max(T_cycle, 0.1)
        # Sample a few cycles at the end for clarity
        n_cycles_show = min(4, int(time[-1] / T_cycle))
        t_start = max(0, time[-1] - n_cycles_show * T_cycle)
        mask = time >= t_start
        t_show = time[mask]
        for j in range(len(t_show) - 1):
            # Determine state from foot vertical velocity (swing = lifting)
            z_vel = (data["target"][leg][mask][j + 1, 2] -
                     data["target"][leg][mask][j, 2]) / (t_show[j + 1] - t_show[j])
            state = "swing" if z_vel > 0.005 else "stance"
            color = leg_colors[leg] if state == "stance" else "#BDC3C7"
            ax.barh(leg, t_show[j + 1] - t_show[j], bar_height,
                    left=t_show[j], color=color, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Gait Phase Diagram — {gname} (color=stance, gray=swing)")
    ax.set_xlim(t_start, time[-1])
    fig.tight_layout()
    fig.savefig(output_dir / f"gait_phase_{gname}.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ gait_phase_{gname}.png")


def main():
    parser = argparse.ArgumentParser(
        description="MyDog — Single Foot Trajectory Simulation"
    )
    parser.add_argument("--leg", default="FR",
                        choices=["FR", "FL", "RR", "RL"],
                        help="Leg to control (default: FR)")
    parser.add_argument("--traj", default="circle",
                        choices=["circle", "line", "sine", "lissajous"],
                        help="Trajectory type (default: circle)")
    parser.add_argument("--urdf", action="store_true",
                        help="Use URDF input (converts to MJCF)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to custom MJCF XML file")
    parser.add_argument("--dt", type=float, default=0.002,
                        help="Simulation time step (default: 0.002)")
    parser.add_argument("--viewer", action="store_true",
                        help="Run with MuJoCo interactive 3D viewer (real-time GUI)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting (headless mode only)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots (default: output/)")

    # ── Gait mode arguments ──
    parser.add_argument("--gait", action="store_true",
                        help="Enable gait-based locomotion (trot/walk/pace/bound)")
    parser.add_argument("--gait-type", default="trot",
                        choices=["trot", "walk", "pace", "bound"],
                        help="Gait pattern (default: trot)")
    parser.add_argument("--gait-T", type=float, default=0.5,
                        help="Gait cycle period in seconds (default: 0.5)")
    parser.add_argument("--gait-duty", type=float, default=0.6,
                        help="Stance duty factor in (0,1) (default: 0.6)")
    parser.add_argument("--step-length", type=float, default=0.06,
                        help="Foot step length in meters (default: 0.06)")
    parser.add_argument("--step-height", type=float, default=0.04,
                        help="Foot lift height in meters (default: 0.04)")
    parser.add_argument("--gait-cycles", type=float, default=5.0,
                        help="Number of gait cycles to simulate in headless "
                             "mode (default: 5)")
    parser.add_argument("--float", action="store_true", dest="floating",
                        help="Use floating-base model (dog runs on ground). "
                             "Only valid with --gait.")
    parser.add_argument("--force", action="store_true",
                        help="Use MIT-style force control (torque via J^T F). "
                             "Only valid with --gait --float. Requires --viewer "
                             "or --headless.")
    parser.add_argument("--target-vx", type=float, default=0.3,
                        help="Target forward velocity for force control (default: 0.3)")
    parser.add_argument("--target-vy", type=float, default=0.0,
                        help="Target lateral velocity for force/MPC control (default: 0.0)")
    parser.add_argument("--target-vyaw", type=float, default=0.0,
                        help="Target yaw rate for force/MPC control in rad/s (default: 0.0)")
    parser.add_argument("--mpc", action="store_true",
                        help="Use SRB convex MPC + MIT impedance control. "
                             "Only valid with --force --gait --float.")
    parser.add_argument("--quintic", action="store_true",
                        help="Use quintic trajectory + static friction force control. "
                             "Only valid with --force --gait --float. "
                             "Mutually exclusive with --mpc.")
    parser.add_argument("--mu-max", type=float, default=0.6,
                        help="Maximum static friction coefficient for "
                             "force distribution (default: 0.6)")
    parser.add_argument("--adapt-params", action="store_true",
                        help="Enable online gait parameter adaptation (Tier 1). "
                             "Only valid with --quintic.")
    parser.add_argument("--momentum", action="store_true",
                        help="Use momentum-based 6×6 Newton-Euler force distribution. "
                             "Only valid with --quintic --force --gait --float.")
    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = prepare_model(use_urdf=args.urdf, with_scene=args.viewer)

    # ── Gait mode ──
    if args.gait:
        gait_type = GaitType(args.gait_type)
        params = GaitParams(
            T_cycle=args.gait_T,
            duty_factor=args.gait_duty,
            step_length=args.step_length,
            step_height=args.step_height,
        )
        # Floating-base: use scene.xml (has ground plane + sky + Go1 freejoint)
        floating = args.floating
        if floating and not args.model:
            model_path = str(SCENE_PATH)

        # ── Force control mode (MIT-style, MPC, or Quintic) ──
        if args.force:
            if not floating:
                print("Error: --force requires --float (floating-base model)")
                sys.exit(1)

            use_mpc = args.mpc
            use_quintic = args.quintic

            if use_mpc and use_quintic:
                print("Error: --quintic and --mpc are mutually exclusive")
                sys.exit(1)

            if args.momentum and not use_quintic:
                print("Error: --momentum requires --quintic")
                sys.exit(1)

            if args.viewer:
                if use_quintic:
                    run_gait_quintic_viewer(model_path, gait_type, params,
                                            dt=args.dt, target_vx=args.target_vx,
                                            target_vy=args.target_vy,
                                            target_vyaw=args.target_vyaw,
                                            mu_max=args.mu_max,
                                            adapt_params=args.adapt_params,
                                            use_momentum=args.momentum)
                elif use_mpc:
                    run_gait_mpc_viewer(model_path, gait_type, params,
                                        dt=args.dt, target_vx=args.target_vx,
                                        target_vy=args.target_vy,
                                        target_vyaw=args.target_vyaw)
                else:
                    run_gait_force_viewer(model_path, gait_type, params,
                                          dt=args.dt, target_vx=args.target_vx,
                                          target_vy=args.target_vy,
                                          target_vyaw=args.target_vyaw)
                return
            else:
                # Headless force mode
                matplotlib.use("Agg")
                output_dir = Path(args.output) if args.output else ROOT / "output"
                if use_quintic:
                    run_gait_quintic_headless(model_path, gait_type, params,
                                              dt=args.dt, total_cycles=args.gait_cycles,
                                              target_vx=args.target_vx,
                                              target_vy=args.target_vy,
                                              target_vyaw=args.target_vyaw,
                                              mu_max=args.mu_max,
                                              adapt_params=args.adapt_params,
                                              use_momentum=args.momentum)
                elif use_mpc:
                    run_gait_mpc_headless(model_path, gait_type, params,
                                          dt=args.dt, total_cycles=args.gait_cycles,
                                          target_vx=args.target_vx,
                                          target_vy=args.target_vy,
                                          target_vyaw=args.target_vyaw)
                else:
                    run_gait_force_headless(model_path, gait_type, params,
                                            dt=args.dt, total_cycles=args.gait_cycles,
                                            target_vx=args.target_vx,
                                            target_vy=args.target_vy,
                                            target_vyaw=args.target_vyaw)
                print("Done.")
                return

        # Viewer mode
        if args.viewer:
            run_gait_simulation_viewer(model_path, gait_type, params,
                                       dt=args.dt, floating=floating)
            return

        # Headless mode
        matplotlib.use("Agg")
        output_dir = Path(args.output) if args.output else ROOT / "output"

        warm_up = 0.5 if floating else 0.2
        data = run_gait_simulation(model_path, gait_type, params,
                                   dt=args.dt, total_cycles=args.gait_cycles,
                                   floating=floating, warm_up=warm_up)

        if not args.no_plot:
            plot_gait_results(data, gait_type, output_dir)

        print("Done.")
        return

    # Viewer mode: interactive 3D GUI
    if args.viewer:
        run_simulation_viewer(model_path, args.leg, args.traj, dt=args.dt)
        return

    # Headless mode: generate plots
    matplotlib.use("Agg")  # non-interactive backend
    output_dir = Path(args.output) if args.output else ROOT / "output"

    # Run simulation
    data = run_simulation(model_path, args.leg, args.traj, dt=args.dt)

    # Plot
    if data is not None and not args.no_plot:
        plot_results(data, args.leg, args.traj, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
