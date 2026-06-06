"""MyDog — Single foot tip trajectory simulation for a robotic dog."""

from .simulator import MuJoCoSim, create_fixed_base_mjcf
from .kinematics import LegKinematics
from .trajectory import (
    LinearTrajectory,
    CircleTrajectory,
    SinusoidalTrajectory,
    LissajousTrajectory,
)
from .controller import IKFootController, create_go1_leg_kinematics
from .urdf_loader import load_urdf_kinematics, urdf_to_mjcf_xml
