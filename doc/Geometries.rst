==================================
 Volume and Projection Geometries
==================================

Tomosipo has six types of geometries with varying degrees of flexibility.

.. tabularcolumns:: |l|p{2px}|
.. list-table:: Geometries
   :width: 50
   :widths: 10 40
   :header-rows: 1

   * - Creation function
     - Geometry
   * - :meth:`tomosipo.parallel`
     - 3D circular parallel beam geometry
   * - :meth:`tomosipo.parallel_vec`
     - 3D arbitrarily-oriented parallel beam geometry
   * - :meth:`tomosipo.cone`
     - 3D circular cone beam geometry
   * - :meth:`tomosipo.cone_vec`
     - 3D arbitrarily-oriented cone beam geometry
   * - :meth:`tomosipo.volume`
     - 3D axis-aligned volume geometry
   * - :meth:`tomosipo.cone_vec`
     - 3D arbitrarily-oriented volume geometry. This object cannot be converted to ASTRA directly, but can be used in geometric computations.



Useful properties
=================

Printed representation
----------------------

.. ipython::
   :verbatim:

   In [13]: import tomosipo as ts
      ....: pg = ts.parallel(angles=3, shape=(10, 15), size=(1, 1.5))
      ....: pg # geometries have a useful representation when printed
   Out[15]: ts.parallel(
       angles=3,
       shape=(10, 15),
       size=(1, 1.5),
   )

Angles, shape, and size
-----------------------

.. ipython::
   :verbatim:

   In [31]: pg.num_angles # number of angles
   Out[31]: 3

   In [18]: pg.angles # Supported for ts.parallel and ts.cone
   Out[18]: array([0.        , 1.04719755, 2.0943951 ])

   In [17]: pg.det_shape # detector shape
   Out[17]: (10, 15)

   In [23]: pg.det_size # detector size (in real-world metrics)
   Out[23]: (1, 1.5)

   In [24]: # In a vector geometry, not all pixels have to be the same size..
      ....: # In that case, detector_sizes can be used to determine the detector size at each angle.
      ....: pg.det_sizes
   Out[24]:
   array([[1. , 1.5],
          [1. , 1.5],
          [1. , 1.5]])

Cone, parallel, vec
-------------------

Determine whether the geometry is a cone beam or parallel beam
geometry and whether or not it is a vector geometry.

.. ipython::
   :verbatim:

   In [34]: pg.is_parallel, pg.is_cone, pg.is_vec
   Out[34]: (True, False, False)


Coordinates for geometric calculations
--------------------------------------

Specific coordinates, such as position (center of detector), `u`, `v`,
corners, detector normal, the lower left corner, etc.

.. ipython::
   :verbatim:

   In [19]: pg.corners
   Out[19]:
   array([[[-0.5       ,  0.        , -0.75      ],
           [ 0.5       ,  0.        , -0.75      ],
           [-0.5       ,  0.        ,  0.75      ],
           [ 0.5       ,  0.        ,  0.75      ]],
          [[-0.5       , -0.64951905, -0.375     ],
           [ 0.5       , -0.64951905, -0.375     ],
           [-0.5       ,  0.64951905,  0.375     ],
           [ 0.5       ,  0.64951905,  0.375     ]],
          [[-0.5       , -0.64951905,  0.375     ],
           [ 0.5       , -0.64951905,  0.375     ],
           [-0.5       ,  0.64951905, -0.375     ],
           [ 0.5       ,  0.64951905, -0.375     ]]])

   In [20]: pg.det_normal
   Out[20]:
   array([[ 0.        ,  0.01      ,  0.        ],
          [ 0.        ,  0.005     , -0.00866025],
          [ 0.        , -0.005     , -0.00866025]])

   In [21]: pg.det_pos
   Out[21]:
   array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]])

   In [25]: pg.det_u
   Out[25]:
   array([[ 0.        ,  0.        ,  0.1       ],
          [ 0.        ,  0.08660254,  0.05      ],
          [ 0.        ,  0.08660254, -0.05      ]])

   In [26]: pg.det_v
   Out[26]:
   array([[0.1, 0. , 0. ],
          [0.1, 0. , 0. ],
          [0.1, 0. , 0. ]])

   In [30]: pg.lower_left_corner
   Out[30]:
   array([[-0.5       ,  0.        , -0.75      ],
          [-0.5       , -0.64951905, -0.375     ],
          [-0.5       , -0.64951905,  0.375     ]])

   In [32]: pg.ray_dir
   Out[32]:
   array([[ 0.       , -1.       ,  0.       ],
          [ 0.       , -0.5      ,  0.8660254],
          [ 0.       ,  0.5      ,  0.8660254]])

   In [33]: pg.src_pos # Only supported on cone and cone_vec geometries


Geometry creation
=================

Circular projection geometries
------------------------------

The following conventions are used:

1. When `size` is not provided, it is taken to be equal to the shape,
   i.e., the detector pixel size is equal to one in each dimension by
   default.
2. The `angles` be provided as a single integer. This is automatically
   expanded to a half circle arc (:meth:`ts.parallel`) or full circle
   arc (:meth:`ts.cone`).
3. An array of `angles` can also be provided, in units of **radians**.
4. The `size` and `shape` parameters can be provided as a single
   float, resulting in a square detector, or as a tuple containing the
   `height` and `width` of the detector.


.. autofunction:: tomosipo.parallel
.. autofunction:: tomosipo.cone

Arbitrary projection geometries
-------------------------------

.. autofunction:: tomosipo.parallel_vec
.. autofunction:: tomosipo.cone_vec

Volume geometries
-----------------

.. autofunction:: tomosipo.volume
.. autofunction:: tomosipo.volume_vec
