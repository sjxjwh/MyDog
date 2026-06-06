#!/usr/bin/env python3
"""Launch MuJoCo interactive viewer for the robot model.

Usage:
    python3 view_model.py                          # Default: go1_fixed.xml
    python3 view_model.py --model model/go1.xml    # Floating base (full body)
    python3 view_model.py --urdf                   # URDF converted model
    python3 view_model.py --animate --leg FR       # Animate trajectory on FR foot
"""

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.simulator import create_fixed_base_mjcf, MuJoCoSim
from src.controller import IKFootController, create_go1_leg_kinematics
from src.trajectory import CircleTrajectory
from src.urdf_loader import urdf_to_mjcf_xml

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare_model(use_urdf: bool, model_path: str | None) -> str:
    """Prepare the model file."""
    if model_path:
        return model_path

    if use_urdf:
        urdf_path = str(ROOT / "model" / "unitree" / "go1.urdf")
        out_path = str(ROOT / "model" / "go1_from_urdf.xml")
        print(f"Converting URDF: {urdf_path} → {out_path}")
        urdf_to_mjcf_xml(urdf_path, out_path, base_link="trunk", fix_base=True)
        return out_path

    fixed_path = ROOT / "model" / "go1_fixed.xml"
    if not fixed_path.exists():
        create_fixed_base_mjcf(
            str(ROOT / "model" / "go1.xml"),
            str(fixed_path),
            base_body="trunk", base_height=0.445
        )
    return str(fixed_path)


def setup_robot(sim: MuJoCoSim, leg: str):
    """Initialize robot to home pose."""
    ctrl = IKFootController(sim, leg, create_go1_leg_kinematics(leg))

    home_angles = [0.0, 0.9, -1.8]
    for jname, angle in zip(ctrl.joint_names, home_angles):
        sim.set_qpos(jname, angle)

    # Set other legs to home too
    for other_leg in ["FL", "RR", "RL"]:
        if other_leg != leg:
            other_ctrl = IKFootController(sim, other_leg, create_go1_leg_kinematics(other_leg))
            for jname, angle in zip(other_ctrl.joint_names, home_angles):
                sim.set_qpos(jname, angle)

    sim.forward()
    return ctrl


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Interactive Viewer")
    parser.add_argument("--model", type=str, default=None, help="Path to MJCF model")
    parser.add_argument("--urdf", action="store_true", help="Use URDF input")
    parser.add_argument("--animate", action="store_true", help="Animate trajectory")
    parser.add_argument("--leg", default="FR", choices=["FR", "FL", "RR", "RL"])
    args = parser.parse_args()

    model_path = prepare_model(args.urdf, args.model)
    print(f"Loading: {model_path}")

    # Load model
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    sim = MuJoCoSim(model_path)
    ctrl = setup_robot(sim, args.leg)

    foot_pos = ctrl.get_foot_position()
    print(f"Model: {mj_model.nq} qpos, {mj_model.nv} qvel, {mj_model.nu} actuators")
    print(f"Foot ({args.leg}) home position: [{foot_pos[0]:.3f}, {foot_pos[1]:.3f}, {foot_pos[2]:.3f}]")

    if args.animate:
        # Pre-compute trajectory
        traj = CircleTrajectory(
            center=foot_pos + np.array([0.0, 0.0, 0.02]),
            radius=0.04, axis="z", T=2.0
        )
        traj_points = traj.sample(dt=0.005)
        n = len(traj_points)
        print(f"Trajectory: {n} waypoints, radius=0.04m")

    print("\nLaunching viewer... (close viewer window to exit)\n")

    step_idx = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Reset to home
        mj_data.qpos[:] = sim.data.qpos[:]
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if args.animate:
                target_world = traj_points[step_idx % n]
                q_target = ctrl.control(target_world)
                if q_target is not None:
                    for jname, angle in zip(ctrl.joint_names, q_target):
                        sim.set_joint_ctrl(
                            ctrl._ctrl_names[ctrl.joint_names.index(jname)],
                            angle
                        )
                step_idx += 1

            sim.step()

            # Sync viewer with simulation state
            mj_data.qpos[:] = sim.data.qpos[:]
            mj_data.qvel[:] = sim.data.qvel[:]
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # ~60 FPS timing
            time_until_next_step = 0.005 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
