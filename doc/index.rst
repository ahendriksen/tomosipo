.. tomosipo documentation master file, created by
   sphinx-quickstart on Tue Jul 21 09:35:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tomosipo: High-performance 3D tomography with flexible geometries in Python
===========================================================================

.. ipython::
   :verbatim:
   :doctest:

   In [10]: import numpy as np
      ....: import tomosipo as ts
      ....: # Create a 3D  parallel-beam geometry collecting 180 projection images over a half-circle.
      ....: # The projection images are 128 pixels high and 192 pixels wide.
      ....: pg = ts.parallel(angles=180, shape=(128, 192))
      ....: # Create a 3D volume geometry on the origin
      ....: vg = ts.volume(shape=128)
      ....: # Create a projection operator
      ....: A = ts.operator(vg, pg)

   In [11]: # Create volume data
      ....: x = np.ones(A.domain_shape, dtype=np.float32)

   In [12]: # Forward project
      ....: y = A(x)

   In [13]: # And backproject the obtained projection data
      ....: bp = A.T(y)

Tomosipo is a pythonic wrapper for the ASTRA-toolbox of
high-performance GPU primitives for 3D tomography.

This library aims to expose a user-friendly API for high-performance
3D tomography, while allowing strict control over resource usage.
In addition, this library enables easy manipulation and visualisation
of 3D geometries.
Finally, the library ingtegrates with deep learning toolkits, such as
PyTorch, and the operator discretization library (ODL) for
optimization in inverse problems.

.. toctree::
   :maxdepth: 2


   readme
   Conventions
   Geometries

   modules
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
