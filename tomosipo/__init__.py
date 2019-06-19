# -*- coding: utf-8 -*-

"""Top-level package for tomosipo."""

__author__ = """Allard Hendriksen"""
__email__ = "allard.hendriksen@cwi.nl"
__version__ = "0.0.1"

from .VolumeGeometry import (
    VolumeGeometry,
    volume,
    volume_from_projection_geometry,
    is_volume_geometry,
)
from . import ProjectionGeometry
from .ConeGeometry import cone
from .GeometryConversion import from_astra_geometry
from .ConeVectorGeometry import cone_vec
from .Operator import forward, backward, fdk, operator
from .display import display_geometry, display_data, display
from .Data import data
from .OrientedBox import OrientedBox
from . import phantom
import warnings

# This is a fundamental constant used for equality checking in
# floating point code.
epsilon = 1e-6

try:
    from astra.experimental import accumulate_FDK
    from astra.experimental import do_composite_BP
    from astra.experimental import do_composite_FP
except AttributeError:
    warnings.warn(
        "Cannot find all required astra.experimental methods. \n"
        "Please make sure you have at least astra version 1.9.x installed. \n"
        "Currently, you can install the astra development version using: \n"
        "> conda install -c astra-toolbox/label/dev astra-toolbox"
    )
