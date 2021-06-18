.. _intro_simple_reconstruction:

A simple reconstruction SIRT reconstruction
===========================================

.. note::
   To keep the size of the documentation in version control manageable, we do
   not include images of the volume and the reconstruction. It is recommended to
   follow along in your favorite python environment so you can see what is going
   on.

In the :ref:`previous tutorial <intro_forward_projection>`, we executed a
forward projection and created a sinogram. Now, we show how to compute a simple
reconstruction.

We continue with the same geometries and projection operator:

.. testcode:: session

   import tomosipo as ts
   import numpy as np

   vg = ts.volume(shape=(32, 32, 32), size=(1, 1, 1))
   pg = ts.parallel(angles=32, shape=(48, 48), size=(1.5, 1.5))
   A = ts.operator(vg, pg)

We create a hollow cube phantom which has an 8 voxel wide border with values
`1.0` and has a center that is zero-valued:

.. testcode:: session
   :skipif: not cuda_available

   x = np.ones(A.domain_shape, dtype=np.float32)
   x[8:-8, 8:-8, 8:-8] = 0.0

And forward project to obtain a sinogram:

.. testcode:: session
   :skipif: not cuda_available

   y = A(x)

Tomosipo does not contain any reconstruction algorithms, but the SIRT algorithm
is quite simple to program yourself.
The algorithm is defined by

.. math::

   x_0 &= \mathbf{0}

   x_{n+1} &= C A^T R (y - A x_n)


with diagonal matrices \\(C\\) and \\(R\\), defined by

.. math::

   C_{jj} &= \frac{1}{\sum_{i} a_{ij}},

   R_{ii} &= \frac{1}{\sum_{j} a_{ij}}.

This is explained in more detail at `Tom Roelandt's blog
<https://tomroelandts.com/articles/the-sirt-algorithm>`_.

First, we prepare preconditioning matrices R and C. Since the matrices are
diagonal, we use vectors of the same shape as x and y (they multiply
element-wise in numpy). Since some values are divided by zero, we clamp the
values using `np.minimum`.

.. testcode:: session
   :skipif: not cuda_available

   R = 1 / A(np.ones(A.domain_shape), dtype=np.float32)
   R = np.minimum(R, 1 / ts.epsilon)
   C = 1 / A.T(np.ones(A.range_shape), dtype=np.float32)
   C = np.minimum(C, 1 / ts.epsilon)

Next, we reconstruct from the sinogram stack `y` into `x_rec`:

.. testcode:: session
   :skipif: not cuda_available

   num_iterations = 50
   x_rec = np.zeros(A.domain_shape, dtype=np.float32)

   for i in range(num_iterations):
       x_rec += C * A.T(R * (y - A(x_rec)))

The result can be shown using matplotlib:

.. testcode:: session
   :skipif: (not cuda_available) or (not matplotlib_available)

   import matplotlib.pyplot as plt
   plt.imshow(x[16, :, :])     # central slice of phantom
   plt.imshow(x_rec[16, :, :]) # central slice of reconstruction


In general, it is recommended to use the `ts_algorithms
<https://github.com/ahendriksen/ts_algorithms>`_ package to compute
reconstructions. This package contains some well-tested reconstruction
algorithms for use with `tomosipo`.

The SIRT algorithm that we just described is not very fast. In fact, the
built-in SIRT algorithm of the ASTRA-toolbox is faster. This is because NumPy
performs computations using the CPU and the ASTRA-toolbox performs the forward
and backprojection on the GPU. Therefore, a intermediate data has to be moved
from and to the GPU. We show how to avoid this in the :ref:`next tutorial
<intro_fast_reconstruction>`.
