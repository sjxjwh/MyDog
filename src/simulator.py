"""MuJoCo simulation wrapper for robot trajectory simulation."""

from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


class MuJoCoSim:
    """High-level wrapper around MuJoCo simulation for a robotic dog."""

    def __init__(self, model_path: str):
        """Load a MuJoCo MJCF model.

        Args:
            model_path: Path to MJCF XML file or scene XML file.
        """
        self.model_path = Path(model_path)
        self._model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self._data = mujoco.MjData(self._model)

        self._body_ids: dict[str, int] = {}
        self._joint_ids: dict[str, int] = {}
        self._actuator_ids: dict[str, int] = {}
        self._site_ids: dict[str, int] = {}

        self._build_name_index()

    def _build_name_index(self):
        """Build lookup dictionaries for body/joint/actuator names."""
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self._body_ids[name] = i

        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._joint_ids[name] = i

        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_ids[name] = i

        for i in range(self._model.nsite):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name:
                self._site_ids[name] = i

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def nq(self) -> int:
        return self._model.nq

    @property
    def nv(self) -> int:
        return self._model.nv

    @property
    def nu(self) -> int:
        return self._model.nu

    def step(self):
        """Advance the simulation by one time step."""
        mujoco.mj_step(self._model, self._data)

    def reset(self, qpos0: Optional[np.ndarray] = None):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        if qpos0 is not None:
            self._data.qpos[:] = qpos0
        mujoco.mj_forward(self._model, self._data)

    def forward(self):
        """Compute forward kinematics (positions, velocities, etc.)."""
        mujoco.mj_forward(self._model, self._data)

    def get_body_position(self, body_name: str) -> np.ndarray:
        """Get world-frame position of a body."""
        bid = self._body_ids.get(body_name)
        if bid is None:
            raise KeyError(f"Body '{body_name}' not found")
        return self._data.xpos[bid].copy()

    def get_body_rotation(self, body_name: str) -> np.ndarray:
        """Get world-frame rotation matrix of a body."""
        bid = self._body_ids.get(body_name)
        if bid is None:
            raise KeyError(f"Body '{body_name}' not found")
        return self._data.xmat[bid].reshape(3, 3).copy()

    def get_joint_qpos(self, joint_name: str) -> float:
        """Get current position of a joint."""
        jid = self._joint_ids.get(joint_name)
        if jid is None:
            raise KeyError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_qposadr[jid]
        return float(self._data.qpos[addr])

    def get_joint_qvel(self, joint_name: str) -> float:
        """Get current velocity of a joint."""
        jid = self._joint_ids.get(joint_name)
        if jid is None:
            raise KeyError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_dofadr[jid]
        return float(self._data.qvel[addr])

    def get_site_position(self, site_name: str) -> np.ndarray:
        """Get world-frame position of a site."""
        sid = self._site_ids.get(site_name)
        if sid is None:
            raise KeyError(f"Site '{site_name}' not found")
        return self._data.site_xpos[sid].copy()

    def set_ctrl(self, ctrl: np.ndarray):
        """Set all actuator control signals."""
        self._data.ctrl[:] = ctrl

    def set_joint_ctrl(self, joint_name: str, value: float):
        """Set control signal for a specific joint (via its actuator)."""
        aid = self._actuator_ids.get(joint_name)
        if aid is not None:
            self._data.ctrl[aid] = value

    def set_qpos(self, joint_name: str, value: float):
        """Set joint position directly (for kinematic setup)."""
        jid = self._joint_ids.get(joint_name)
        if jid is None:
            raise KeyError(f"Joint '{joint_name}' not found")
        addr = self._model.jnt_qposadr[jid]
        self._data.qpos[addr] = value

    def get_foot_site(self, leg_prefix: str) -> str:
        """Get the foot site name for a given leg prefix (e.g., 'FR').

        In the Go1 MJCF model, foot sites are named 'FR', 'FL', 'RR', 'RL'.
        """
        if leg_prefix in self._site_ids:
            return leg_prefix
        # Try alternate naming
        candidates = [
            leg_prefix,
            f"{leg_prefix}_foot",
        ]
        for c in candidates:
            if c in self._site_ids:
                return c
        raise KeyError(f"No foot site found for leg '{leg_prefix}'")

    def get_foot_position(self, leg_prefix: str) -> np.ndarray:
        """Get world-frame position of a foot."""
        site = self.get_foot_site(leg_prefix)
        return self.get_site_position(site)

    def get_all_joint_positions(self) -> dict[str, float]:
        """Get current positions of all joints."""
        result = {}
        for name, jid in self._joint_ids.items():
            addr = self._model.jnt_qposadr[jid]
            result[name] = float(self._data.qpos[addr])
        return result

    def joint_names(self) -> list[str]:
        """Return list of all joint names."""
        return list(self._joint_ids.keys())

    def actuator_names(self) -> list[str]:
        """Return list of actuator names."""
        return list(self._actuator_ids.keys())


def create_fixed_base_mjcf(mjcf_path: str, output_path: str,
                           base_body: str = "trunk",
                           base_height: float = 0.445) -> str:
    """Create a fixed-base variant of an MJCF model by removing the freejoint.

    Removes the <freejoint/> from the base body and fixes it at the specified height.

    Args:
        mjcf_path: Path to the original MJCF file.
        output_path: Path to write the fixed-base MJCF.
        base_body: Name of the body that has the freejoint.
        base_height: Height at which to fix the base (default: 0.445 for Go1).

    Returns:
        Path to the fixed MJCF file.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no <worldbody>")

    # Find the base body with freejoint and remove the freejoint
    for body in worldbody.findall(".//body"):
        if body.get("name") == base_body:
            # Remove <freejoint/>
            for fj in body.findall("freejoint"):
                body.remove(fj)
            # Set body position to be fixed at base_height
            current_pos = body.get("pos", "0 0 0")
            parts = current_pos.split()
            while len(parts) < 3:
                parts.append("0")
            parts[2] = str(base_height)
            body.set("pos", " ".join(parts))
            break

    # Fix keyframes: remove the 7 freejoint qpos values (3 pos + 4 quat)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        for kf in keyframe.findall("key"):
            qpos_str = kf.get("qpos", "")
            qpos_vals = qpos_str.split()
            # Freejoint has 7 qpos values; joints have 12
            if len(qpos_vals) >= 7:
                kf.set("qpos", " ".join(qpos_vals[7:]))

    # Lighten the 'dark' visual material for better rendering visibility
    asset = root.find("asset")
    if asset is not None:
        for mat in asset.findall("material"):
            if mat.get("name") == "dark":
                mat.set("rgba", "0.5 0.5 0.55 1")
                mat.set("specular", "0.3")
                mat.set("shininess", "0.4")

    # Write
    xml_str = ET.tostring(root, encoding="unicode")
    with open(output_path, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(xml_str)

    return output_path
