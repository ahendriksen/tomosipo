""" This module provides a short-hand to add cupy support to tomosipo

To enable support for torch tensors in tomosipo, use:

>>> import tomosipo.cupy

"""
# This import is needed to enable to cupy linking backend.
import tomosipo.links.cupy
