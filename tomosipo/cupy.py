""" This module can be used to force interoperability with cupy

In general it is not necessary to explicitly import this module.

"""
import warnings

try:
    import cupy
except ModuleNotFoundError:
    warnings.warn(
        "\n------------------------------------------------------------\n\n"
        "Cannot import cupy package. \n"
        "Please make sure to install cupy. \n"
        "You can install cupy using: \n\n"
        " > conda install cupy -c conda-forge \n"
        "\n------------------------------------------------------------\n\n"
    )
    raise

# This import is needed to enable to cupy linking backend.
import tomosipo.links.cupy
