API Summary
===========

We summarize important functions and classes that you may encounter in daily
usage of tomosipo. This list is not meant to be exhaustive.

.. _summary-create-geometries:
Create geometries
-----------------

.. autosummary::
   :toctree: _autosummary

   tomosipo.volume
   tomosipo.parallel
   tomosipo.cone
   tomosipo.volume_vec
   tomosipo.parallel_vec
   tomosipo.cone_vec

Create geometric transforms
---------------------------

.. autosummary::
   :toctree: _autosummary

   tomosipo.translate
   tomosipo.rotate
   tomosipo.scale
   tomosipo.to_perspective
   tomosipo.from_perspective

Create projection operator
--------------------------

.. autosummary::
   :toctree: _autosummary

   tomosipo.operator
   tomosipo.Operator.Operator

Display geometries
------------------

.. autosummary::
   :toctree: _autosummary

   tomosipo.svg

.. _summary-geometry-classes:

Geometry Classes
----------------

.. autosummary::
   :toctree: _autosummary

   tomosipo.geometry.VolumeGeometry
   tomosipo.geometry.VolumeVectorGeometry
   tomosipo.geometry.ProjectionGeometry
   tomosipo.geometry.ParallelGeometry
   tomosipo.geometry.ParallelVectorGeometry
   tomosipo.geometry.ConeGeometry
   tomosipo.geometry.ConeVectorGeometry
   tomosipo.geometry.Transform

Interoperability
----------------

ASTRA-toolbox
^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary

   tomosipo.to_astra
   tomosipo.from_astra

ODL
^^^

.. autosummary::
   :toctree: _autosummary

   tomosipo.odl.from_odl

Cupy
^^^^

.. autosummary::
   :toctree: _autosummary

   tomosipo.cupy
