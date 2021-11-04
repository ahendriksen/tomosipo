# -*- coding: utf-8 -*-

"""Top-level package for tomosipo."""

__author__ = """Allard Hendriksen"""
__email__ = "allard.hendriksen@cwi.nl"
# Also edit the version in setup.py!
__version__ = "0.4.1"

from .Operator import operator
from .Data import data
from .geometry.volume import volume
from .geometry.volume_vec import volume_vec
from . import geometry
from .geometry.cone import cone
from .geometry.cone_vec import cone_vec
from .geometry.parallel_vec import parallel_vec
from .geometry.parallel import parallel
from .geometry.transform import (
    translate,
    scale,
    rotate,
    reflect,
    to_perspective,
    from_perspective,
)
from .geometry.concatenate import concatenate

from .links.base import link
from .astra import (
    from_astra,
    to_astra,
)
from .svg import svg

from . import phantom
from . import types

# This is a fundamental constant used for equality checking in
# floating point code.
epsilon = 1e-8

try:

    def __import_astra_functionality():
        import astra.experimental
        from astra.experimental import accumulate_FDK
        from astra.experimental import do_composite
        from astra.experimental import direct_FPBP3D

    __import_astra_functionality()
except AttributeError:
    raise ImportError(
        "Cannot find all required astra.experimental methods. \n"
        "Please make sure you have at least ASTRA version 2.0 installed. \n"
        "You can install the latest ASTRA version using: \n"
        "> conda install astra-toolbox=2.0 -c astra-toolbox "
    )
except ImportError:
    raise ImportError(
        "Cannot find all required astra.experimental methods. \n"
        "Please make sure you have at least ASTRA version 2.0 installed. \n"
        "You can install the latest ASTRA version using: \n"
        "> conda install astra-toolbox=2.0 -c astra-toolbox "
    )
