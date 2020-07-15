# -*- coding: utf-8 -*-

"""Top-level package for tomosipo."""

__author__ = """Allard Hendriksen"""
__email__ = "allard.hendriksen@cwi.nl"
__version__ = "0.0.1"

from .Operator import forward, backward, fdk, operator
from .Data import data
from .geometry.volume import volume, volume_from_projection_geometry
from . import geometry
from .geometry.cone import cone
from .geometry.cone_vec import cone_vec
from .geometry.parallel_vec import parallel_vec
from .geometry.parallel import parallel
from .geometry.oriented_box import box
from .geometry.transform import (
    translate,
    scale,
    rotate,
    to_perspective,
    from_perspective,
)
from .geometry.concatenate import concatenate

from .links.base import link
from .astra import (
    from_astra,
    to_astra,
)



from . import phantom
import warnings

# This is a fundamental constant used for equality checking in
# floating point code.
epsilon = 1e-8

try:
    from astra.experimental import accumulate_FDK
    from astra.experimental import do_composite
    from astra.experimental import direct_FPBP3D
except AttributeError:
    warnings.warn(
        "Cannot find all required astra.experimental methods. \n"
        "Please make sure you have at least ASTRA version 1.9.9-dev4 installed. \n"
        "You can install the latest ASTRA development version using: \n"
        "> conda install astra-toolbox -c astra-toolbox/label/dev "
    )
