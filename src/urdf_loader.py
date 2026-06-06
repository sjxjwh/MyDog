"""URDF to MuJoCo MJCF converter.

Parses a URDF file and converts it into a MuJoCo-compatible MJCF XML model.
Handles links, revolute/continuous/prismatic/fixed joints, collision and visual
geometries, inertial parameters, and mesh references.
"""

from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np


def _parse_origin(elem) -> tuple:
    """Parse <origin> element, returning (xyz, rpy)."""
    origin = elem.find("origin")
    if origin is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array(
        [float(x) for x in origin.get("xyz", "0 0 0").split()]
    )
    rpy = np.array(
        [float(x) for x in origin.get("rpy", "0 0 0").split()]
    )
    return xyz, rpy


def _rpy_to_euler_mat(rpy):
    """Convert roll-pitch-yaw to rotation matrix (intrinsic ZYX)."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx  # intrinsic ZYX = matmul Rz Ry Rx


def _mat_to_quat(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    trace = np.clip(trace, -1.0, 3.0)  # numerical safety

    if trace > 0:
        w = np.sqrt(1.0 + trace) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    else:
        # trace <= 0: find largest diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0)) / 2.0
            w = (R[2, 1] - R[1, 2]) / (4.0 * max(x, 1e-16))
            y = (R[0, 1] + R[1, 0]) / (4.0 * max(x, 1e-16))
            z = (R[2, 0] + R[0, 2]) / (4.0 * max(x, 1e-16))
        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0)) / 2.0
            w = (R[0, 2] - R[2, 0]) / (4.0 * max(y, 1e-16))
            x = (R[0, 1] + R[1, 0]) / (4.0 * max(y, 1e-16))
            z = (R[1, 2] + R[2, 1]) / (4.0 * max(y, 1e-16))
        else:
            z = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0)) / 2.0
            w = (R[1, 0] - R[0, 1]) / (4.0 * max(z, 1e-16))
            x = (R[2, 0] + R[0, 2]) / (4.0 * max(z, 1e-16))
            y = (R[1, 2] + R[2, 1]) / (4.0 * max(z, 1e-16))

    # Normalize
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 1e-16:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    return np.array([w, x, y, z])


def _parse_axis(elem, default=None):
    """Parse <axis> element, returning numpy array."""
    if default is None:
        default = np.array([0, 1, 0])
    axis_elem = elem.find("axis")
    if axis_elem is None:
        return default
    return np.array([float(x) for x in axis_elem.get("xyz", "0 1 0").split()])


def _parse_limit(elem):
    """Parse <limit> element, returning (lower, upper, effort, velocity)."""
    limit = elem.find("limit")
    if limit is None:
        return -np.inf, np.inf, 0.0, 0.0
    return (
        float(limit.get("lower", "-inf")),
        float(limit.get("upper", "+inf")),
        float(limit.get("effort", "0")),
        float(limit.get("velocity", "0")),
    )


def _parse_dynamics(elem):
    """Parse <dynamics> element, returning (damping, friction)."""
    dyn = elem.find("dynamics")
    if dyn is None:
        return 0.0, 0.0
    return float(dyn.get("damping", "0")), float(dyn.get("friction", "0"))


def _parse_inertial(link_elem) -> Optional[dict]:
    """Parse <inertial> element of a link."""
    inertial = link_elem.find("inertial")
    if inertial is None:
        return None

    mass_elem = inertial.find("mass")
    inertia_elem = inertial.find("inertia")
    if mass_elem is None or inertia_elem is None:
        return None

    origin_xyz, origin_rpy = _parse_origin(inertial)
    return {
        "mass": float(mass_elem.get("value", "0")),
        "ixx": float(inertia_elem.get("ixx", "0")),
        "ixy": float(inertia_elem.get("ixy", "0")),
        "ixz": float(inertia_elem.get("ixz", "0")),
        "iyy": float(inertia_elem.get("iyy", "0")),
        "iyz": float(inertia_elem.get("iyz", "0")),
        "izz": float(inertia_elem.get("izz", "0")),
        "origin_xyz": origin_xyz,
        "origin_rpy": origin_rpy,
    }


def _parse_geometry(geom_elem) -> Optional[dict]:
    """Parse a <geometry> element (box, cylinder, sphere, mesh)."""
    if geom_elem is None:
        return None
    geo = geom_elem.find("geometry")
    if geo is None:
        return None

    for gtype in ("box", "cylinder", "sphere", "mesh"):
        g = geo.find(gtype)
        if g is not None:
            if gtype == "box":
                size = np.array([float(s) for s in g.get("size", "0 0 0").split()])
                # URDF box size is full extents; MJCF uses half-extents
                return {"type": "box", "size": size / 2.0}
            elif gtype == "cylinder":
                r = float(g.get("radius", "0"))
                l = float(g.get("length", "0"))
                return {"type": "cylinder", "size": np.array([r, l / 2.0])}
            elif gtype == "sphere":
                r = float(g.get("radius", "0"))
                return {"type": "sphere", "size": np.array([r])}
            elif gtype == "mesh":
                filename = g.get("filename", "")
                scale = g.get("scale", "1 1 1")
                return {
                    "type": "mesh",
                    "filename": filename,
                    "scale": np.array([float(s) for s in scale.split()]),
                }
    return None


def _geom_to_mjcf_str(geom_info, pos=None, quat=None, rgba=None, geom_type_override=None,
                      mesh_name_map=None):
    """Convert a parsed geometry dict to an MJCF <geom> string.

    Args:
        geom_info: Parsed geometry dict from _parse_geometry.
        pos, quat, rgba: Override position, orientation, color.
        geom_type_override: Override the geometry type.
        mesh_name_map: Optional dict mapping original mesh filenames to asset names.
    """
    if geom_info is None:
        return ""

    gtype = geom_type_override or geom_info["type"]
    size = geom_info["size"]
    size_str = " ".join(f"{s:.6g}" for s in size)

    attrs = [f'type="{gtype}"']
    if gtype == "mesh":
        mesh_ref = geom_info["filename"]
        if mesh_name_map and mesh_ref in mesh_name_map:
            mesh_ref = mesh_name_map[mesh_ref]
        else:
            # Use just the stem as the mesh name
            mesh_ref = Path(geom_info["filename"]).stem
        attrs.append(f'mesh="{mesh_ref}"')
        if not np.allclose(geom_info.get("scale", [1, 1, 1]), [1, 1, 1]):
            scale_str = " ".join(f"{s:.6g}" for s in geom_info["scale"])
            attrs.append(f'scale="{scale_str}"')
    else:
        attrs.append(f'size="{size_str}"')

    if pos is not None and not np.allclose(pos, 0):
        attrs.append(f'pos="{" ".join(f"{p:.6g}" for p in pos)}"')
    if quat is not None and not np.allclose(quat, [1, 0, 0, 0]):
        attrs.append(f'quat="{" ".join(f"{q:.6g}" for q in quat)}"')
    if rgba is not None:
        attrs.append(f'rgba="{" ".join(f"{c:.4g}" for c in rgba)}"')

    return "<geom " + " ".join(attrs) + "/>"


def load_urdf_kinematics(urdf_path: str) -> dict:
    """Parse URDF and return a kinematic tree structure.

    Returns a dict with:
      - links: dict[name] = {mass, parent, children, joint_info, inertial, collision, visual}
      - root_link: name of the root link
      - joint_names: ordered list of movable joint names
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    joint_info = {}

    # First pass: collect all links
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        inertial = _parse_inertial(link_elem)

        collision_geom = None
        collision_elem = link_elem.find("collision")
        if collision_elem is not None:
            collision_geom = _parse_geometry(collision_elem)

        visual_geom = None
        visual_elem = link_elem.find("visual")
        if visual_elem is not None:
            visual_geom = _parse_geometry(visual_elem)

        links[name] = {
            "inertial": inertial,
            "collision": collision_geom,
            "visual": visual_geom,
            "children": [],
            "parent": None,
            "joint": None,
        }

    # Second pass: build tree from joints
    movable_joints = []
    for joint_elem in root.findall("joint"):
        jname = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")

        origin_xyz, origin_rpy = _parse_origin(joint_elem)
        axis = _parse_axis(joint_elem)
        lower, upper, effort, velocity = _parse_limit(joint_elem)
        damping, friction = _parse_dynamics(joint_elem)

        R_origin = _rpy_to_euler_mat(origin_rpy)
        quat = _mat_to_quat(R_origin)

        joint_data = {
            "name": jname,
            "type": jtype,
            "parent": parent,
            "child": child,
            "origin_xyz": origin_xyz,
            "origin_rpy": origin_rpy,
            "origin_quat": quat,
            "axis": axis,
            "lower": lower,
            "upper": upper,
            "effort": effort,
            "velocity": velocity,
            "damping": damping,
            "friction": friction,
        }

        if child in links:
            links[child]["parent"] = parent
            links[child]["joint"] = joint_data
            if parent in links:
                links[parent]["children"].append(child)

        if jtype in ("revolute", "continuous", "prismatic"):
            movable_joints.append(jname)
            joint_info[jname] = joint_data

    # Find root link
    root_link = None
    for name, info in links.items():
        if info["parent"] is None and len(info["children"]) > 0:
            root_link = name
            break
    if root_link is None:
        root_link = next(iter(links.keys()))

    return {
        "links": links,
        "root_link": root_link,
        "movable_joints": movable_joints,
        "joint_info": joint_info,
    }


def urdf_to_mjcf_xml(urdf_path: str, output_path: Optional[str] = None,
                     base_link: str = "trunk", fix_base: bool = True) -> str:
    """Convert a URDF file to MuJoCo MJCF XML string.

    Args:
        urdf_path: Path to the URDF file.
        output_path: If provided, write MJCF to this file. Mesh paths in the output
                     are adjusted to be relative to the output file's directory.
        base_link: Name of the body to use as the root (skips ancestors).
        fix_base: If True, weld the base link to the world instead of adding a free joint.

    Returns:
        MJCF XML string.
    """
    urdf_path = Path(urdf_path).resolve()
    urdf_dir = urdf_path.parent

    if output_path:
        output_path = Path(output_path).resolve()
        output_dir = output_path.parent
    else:
        output_dir = urdf_dir

    kin = load_urdf_kinematics(str(urdf_path))
    links = kin["links"]

    # Build MJCF XML lines
    lines = ['<mujoco model="converted">']
    lines.append('  <compiler angle="radian" autolimits="true"/>')
    lines.append("")
    lines.append("  <default>")
    lines.append('    <geom rgba="0.3 0.3 0.3 1" friction="0.8 0.02 0.01"/>')
    lines.append('    <joint damping="1" armature="0.01"/>')
    lines.append("  </default>")
    lines.append("")

    # Collect meshes for asset section, with paths relative to output directory
    meshes_seen = {}

    def _resolve_mesh_path(filename: str) -> Path:
        """Resolve a mesh filename (may include package:// URI) to an absolute path."""
        fname = filename
        # Handle ROS package:// URIs
        if fname.startswith("package://"):
            # Extract package name and relative path
            # package://go1_description/meshes/hip.dae → meshes/hip.dae
            parts = fname.replace("package://", "").split("/", 1)
            if len(parts) == 2:
                pkg_name, rel_path = parts
                # Map package name to directory
                # go1_description → model/unitree/
                pkg_dir = urdf_dir  # The URDF lives in the package root
                fname = str(pkg_dir / rel_path)
        return Path(fname)

    # MuJoCo-compatible mesh extensions
    MUJOCO_MESH_EXTS = {".stl", ".obj"}

    def collect_meshes(link_name):
        if link_name not in links:
            return
        info = links[link_name]
        for geom_key in ("visual", "collision"):
            g = info.get(geom_key)
            if g and g["type"] == "mesh":
                mesh_path = _resolve_mesh_path(g["filename"])
                mesh_abs = mesh_path.resolve()
                # Skip non-MuJoCo-compatible mesh files (e.g., DAE)
                if mesh_abs.suffix.lower() not in MUJOCO_MESH_EXTS:
                    continue
                # Compute path relative to output directory
                try:
                    mesh_rel = mesh_abs.relative_to(output_dir)
                except ValueError:
                    mesh_rel = mesh_abs
                mesh_key = str(mesh_rel)
                if mesh_key not in meshes_seen:
                    meshes_seen[mesh_key] = mesh_abs

    # Collect all meshes
    for link_name in links:
        collect_meshes(link_name)

    if meshes_seen:
        lines.append("  <asset>")
        for mesh_rel in sorted(meshes_seen.keys()):
            # Use the relative path as both the mesh name (for referencing) and file
            mesh_name = Path(mesh_rel).stem
            lines.append(f'    <mesh name="{mesh_name}" file="{mesh_rel}"/>')
        lines.append("  </asset>")
        lines.append("")

    lines.append("  <worldbody>")

    # Recursively build body tree starting from base_link
    visited = set()

    def write_body(link_name, indent=4):
        if link_name in visited or link_name not in links:
            return
        visited.add(link_name)

        info = links[link_name]
        prefix = " " * indent
        joint = info.get("joint")

        # Determine body position from joint origin
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if joint is not None:
            pos = joint["origin_xyz"]
            quat = joint["origin_quat"]

        lines.append(f'{prefix}<body name="{link_name}" pos="{" ".join(f"{p:.6g}" for p in pos)}"'
                     f' quat="{" ".join(f"{q:.6g}" for q in quat)}">')

        # Inertial
        if info["inertial"]:
            i = info["inertial"]
            lines.append(
                f'{prefix}  <inertial pos="{" ".join(f"{p:.6g}" for p in i["origin_xyz"])}" '
                f'mass="{i["mass"]:.6g}" '
                f'diaginertia="{i["ixx"]:.6g} {i["iyy"]:.6g} {i["izz"]:.6g}"/>'
            )

        # Joint
        if joint is not None:
            jtype = joint["type"]
            if jtype == "revolute":
                mj_type = "hinge"
            elif jtype == "continuous":
                mj_type = "hinge"
            elif jtype == "prismatic":
                mj_type = "slide"
            elif jtype == "fixed":
                mj_type = None  # skip joint
            else:
                mj_type = None

            if mj_type is not None:
                axis_str = " ".join(f"{a:.6g}" for a in joint["axis"])
                attrs = [f'name="{joint["name"]}"', f'type="{mj_type}"', f'axis="{axis_str}"']
                if joint["damping"] > 0:
                    attrs.append(f'damping="{joint["damping"]:.4g}"')
                if joint["friction"] > 0:
                    attrs.append(f'frictionloss="{joint["friction"]:.4g}"')
                if np.isfinite(joint["lower"]) and np.isfinite(joint["upper"]):
                    attrs.append(f'range="{joint["lower"]:.6g} {joint["upper"]:.6g}"')
                lines.append(f"{prefix}  <joint {' '.join(attrs)}/>")

        # Collision geometry
        if info["collision"]:
            g = info["collision"]
            pos_override = np.zeros(3)
            quat_override = np.array([1.0, 0.0, 0.0, 0.0])
            lines.append(f"{prefix}  " + _geom_to_mjcf_str(
                g, pos=pos_override, quat=quat_override,
                rgba=np.array([0.6, 0.6, 0.6, 0.5])
            ))
            # If mesh, add collision
            if g["type"] == "mesh":
                lines.append(f"{prefix}  " + _geom_to_mjcf_str(
                    g, pos=pos_override, quat=quat_override,
                    rgba=np.array([0.6, 0.6, 0.6, 0.5]), geom_type_override="mesh"
                ))

        # Visual geometry (comment out for now, collision is sufficient)
        # if info["visual"]:
        #     g = info["visual"]
        #     # ...similar

        # Add foot site for foot links
        if link_name.endswith("_foot") or "foot" in link_name.lower():
            foot_short = link_name.split("_")[0]  # e.g., FR_foot → FR
            lines.append(f'{prefix}  <site name="{foot_short}" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>')

        # Recurse children
        for child_name in info["children"]:
            write_body(child_name, indent + 2)

        lines.append(f"{prefix}</body>")

    # Start from base_link
    if fix_base:
        # Add a base body that's fixed to the world
        # Find the joint from the trunk's parent
        trunk_info = links.get(base_link, {})
        trunk_joint = trunk_info.get("joint")
        trunk_pos = np.array([0, 0, 0])
        trunk_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if trunk_joint is not None:
            trunk_pos = trunk_joint["origin_xyz"]
            trunk_quat = trunk_joint["origin_quat"]

        # For Go1, the base link (trunk) is at z ≈ 0.445 from the world
        lines.append(
            f'    <body name="world_base" pos="{" ".join(f"{p:.6g}" for p in trunk_pos)}"'
            f' quat="{" ".join(f"{q:.6g}" for q in trunk_quat)}">'
        )
        write_body(base_link, indent=6)
        lines.append("    </body>")
    else:
        # Add free joint
        lines.append(f'    <body name="{base_link}">')
        lines.append('      <freejoint/>')
        write_body(base_link, indent=6)
        lines.append("    </body>")

    lines.append("  </worldbody>")
    lines.append("")

    # Actuators: one position actuator per revolute joint
    actuators = []
    for jname in kin["movable_joints"]:
        if "rotor" not in jname:  # skip rotor dummy joints
            actuators.append(f'    <position name="{jname}" joint="{jname}"/>')

    if actuators:
        lines.append("  <actuator>")
        lines.extend(actuators)
        lines.append("  </actuator>")

    lines.append("</mujoco>")

    mjcf_str = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(mjcf_str)

    return mjcf_str
