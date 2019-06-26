# -*- coding: utf-8 -*-

"""Top-level package for tomosipo."""

__author__ = """Allard Hendriksen"""
__email__ = "allard.hendriksen@cwi.nl"
__version__ = "0.0.1"

from .display import display
from .Operator import forward, backward, fdk, operator
from .Data import data
from .geometry.volume import volume, volume_from_projection_geometry
from . import geometry
from .geometry.cone import cone
from .geometry.conversion import from_astra_geometry
from .geometry.cone_vec import cone_vec
from .geometry.oriented_box import box
from .geometry.transform import (
    identity,
    translate,
    scale,
    rotate,
    to_perspective,
    from_perspective,
)

from . import phantom
import warnings

# This is a fundamental constant used for equality checking in
# floating point code.
epsilon = 1e-8

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
