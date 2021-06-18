Editable helical cone beam template
===================================

The following can serve as a template to construct a flexible helical cone beam
geometry. For a step by step walk through about modeling geometries in the lab
frame, see :ref:`intro_lab_frame`.

.. testcode::

   import tomosipo as ts
   import numpy as np

   # Detector parameters:
   pixel_size = 2.0
   detector_shape = (1, 1)
   source_position = (0, -3, 0)
   detector_position = (0, 2, 0)

   # Volume parameters:
   voxel_size = 1.0
   volume_shape = np.array([1, 1, 1])
   volume_start_pos = (-1, 0, 0)

   # Helical parameters:
   num_time_steps = 20
   rot_axis_pos = (0, 0.0, 0.0)
   t = np.linspace(0, 1, num_time_steps) # time
   angles = 4 * np.pi * t                # angle
   h = 2.0                               # Vertical "speed"

   # Geometries
   pg = ts.cone_vec(
       shape=detector_shape,
       src_pos=source_position,
       det_pos=detector_position,
       det_v=(pixel_size, 0, 0), # points up
       det_u=(0, 0, pixel_size), # x-axis, points along detector width
   )

   vg0 = ts.volume(
       shape=volume_shape,
       pos=volume_start_pos,
       size=volume_shape * voxel_size,
   )

   R = ts.rotate(pos=rot_axis_pos, axis=(1, 0, 0), angles=angles)
   T = ts.translate(axis=(1, 0, 0), alpha = h * t)

   vg = T * R * vg0.to_vec()

   A = ts.operator(vg, pg)
