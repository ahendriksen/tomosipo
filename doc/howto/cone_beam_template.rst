Editable cone beam template
===============================

The following can serve as a template to construct a flexible cone beam
geometry. For a step by step walk through about modeling geometries in the lab
frame, see :ref:`intro_lab_frame`.

.. testcode::

   import tomosipo as ts
   import numpy as np

   # Detector parameters:
   pixel_size = 1.0
   detector_shape = (10, 10)
   source_position = (0, -10, 0)
   detector_position = (0, 2, 0)

   # Volume parameters:
   volume_shape = np.array([1, 1, 1])
   voxel_size = np.array([1.0, 1.0, 1.0])

   # Rotation parameters
   num_angles = 10
   angles = np.linspace(0, np.pi, num_angles, endpoint=False)
   rot_axis_pos = (0, 0.1, 0.2)

   # Geometries:
   pg = ts.cone_vec(
       shape=detector_shape,
       src_pos=source_position,
       det_pos=detector_position,
       det_v=(pixel_size, 0, 0), # points up
       det_u=(0, 0, pixel_size), # x-axis, points along detector width
   )

   vg0 = ts.volume(
       shape=volume_shape,
       pos=(0, 0, 0),
       size=volume_shape * voxel_size,
   )
   R = ts.rotate(pos=rot_axis_pos, axis=(1, 0, 0), angles=angles)
   vg = R * vg0.to_vec()

   A = ts.operator(vg, pg)
