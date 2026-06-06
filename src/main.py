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

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Default model paths
MJCF_PATH = ROOT / "model" / "go1.xml"
MJCF_FIXED_PATH = ROOT / "model" / "go1_fixed.xml"
SCENE_FIXED_PATH = ROOT / "model" / "scene_fixed.xml"
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
    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = prepare_model(use_urdf=args.urdf, with_scene=args.viewer)

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
