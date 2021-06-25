.. _topics_operator:

Tomographic projection operator
===============================

The tomographic projection operator is used to actually perform computations.

Creating an operator
--------------------

The :py:meth:`ts.operator <tomosipo.Operator.operator>` function creates a
tomographic projection operator from a volume and projection geometry.

.. testcode:: operator

    import tomosipo as ts
    vg = ts.volume(shape=32)
    pg = ts.parallel(angles=48, shape=48)
    A = ts.operator(vg, pg)

In general, you can combine any volume geometry and projection geometry. If both
the volume and projection geometry are moving in time, then the number of steps
in their movement must be equal.

Data shapes and geometries
--------------------------

To obtain the expected shape of the input and output data of the operator, use
the :py:attr:`domain_shape <tomosipo.Operator.Operator.domain_shape>` and
:py:attr:`range_shape <tomosipo.Operator.Operator.range_shape>` properties.

.. doctest:: operator

   >>> A.domain_shape
   (32, 32, 32)
   >>> A.range_shape
   (48, 48, 48)

To obtain the geometries, use the :py:attr:`domain
<tomosipo.Operator.Operator.domain>` and :py:attr:`range
<tomosipo.Operator.Operator.range>` properties. These are named in this way ---
rather than `volume_geometry` and `projection_geometry` --- to be consistent
with the transpose. We have

.. doctest:: operator

   >>> A.domain == A.T.range # , and
   True
   >>> A.range == A.T.domain
   True

Computing a projection
----------------------

The forward projection and backprojection can be computed as follows:

.. testcode:: operator
   :skipif: not cuda_available

   import numpy as np
   x = np.ones(A.domain_shape, dtype=np.float32)
   y = A(x)
   bp = A.T(y)

Note that we create a numpy array containing float32 values. The ASTRA-toolbox,
which executes the projection, only support float32 values. This is described in
more detail in :ref:`topics_operator_data_handling`.

Supersampling
-------------

In addition to the geometry parameters, :py:meth:`ts.operator
<tomosipo.Operator.operator>` has the `detector_supersampling` and
`voxel_supersampling`. These can be used when the voxel and pixel resolution are
different, or when aliasing is a concern.

The `voxel_supersampling` argument is used in the **backprojection**. Suppose it
equals `d`. If `d` equals 1 (the default), then rays are cast through the center
of the voxel onto the detector. Otherwise, $d^3$ rays are cast through each
voxel and the average of the backprojection in each of these locations is
calculated.

The `pixel_supersampling` argument is used in the **projection**. Suppose it
equals `d`. If `d` equals 1 (the default), then rays are cast onto the center of
each detector pixel. Otherwise, $d^2$ rays are cast onto each detector pixel and
the average of the projection onto each of these location is calculated.


Mutability and additivity
-------------------------

If you have a pre-allocated array, then it may make sense to use the `out`
parameter:

.. testcode:: operator
   :skipif: not cuda_available

   y = np.zeros(A.range_shape, dtype=np.float32)
   A(x, out=y)

This overwrites submitted array `out=y`.

In most situations, using the `out` parameter is not really necessary. In
extreme memory-limited situations, it may make sense to use it when you have an
"additive" operator. An additive operator can be created as follows:

.. doctest:: operator

   >>> A_additive = ts.operator(vg, pg, additive=True)

An additive operator does not overwrite the `out` argument, but *adds* to it.
In the code below, the start with all ones on the detector.
First, we add the forward projection of `x` to it using the normal immutable style.
Then, we update `y0` variable in place and add the forward projection of `x`.

.. doctest:: operator
   :skipif: not cuda_available

   >>> y0 = np.ones(A.range_shape, dtype=np.float32)
   >>> y = y0 + A(x)                        # compute y "normally"
   >>> y_additive = A_additive(x, out=y0)   # mutate in place
   >>> assert y_additive is y0
   >>> np.allclose(y_additive, y)
   True

This style of programming saves memory. In general, the performance benefits
tend to be marginal.

.. _topics_operator_data_handling:

Data handling
-------------

Foreign array support
^^^^^^^^^^^^^^^^^^^^^

The :py:class:`Operator <tomosipo.Operator.Operator>` class can handle many
types of arrays. So far, we have seen it operate on NumPy arrays. In addition,
there is support for operating on CuPy and PyTorch arrays.

When the PyTorch package is installed, one can create torch arrays and these
will be handled by tomosipo:

.. doctest:: operator
   :skipif: not cuda_available or not torch_available

   >>> import torch
   >>> x = torch.ones(A.domain_shape)
   >>> y = A(x)
   >>> type(y)
   <class 'torch.Tensor'>
   >>> y.dtype
   torch.float32

As you can see, the input and output array type match. If the input is a NumPy
array, then the output is a NumPy array as well. Likewise, If the input is a PyTorch
tensor, then the output is a PyTorch tensor.


Array device preservation
^^^^^^^^^^^^^^^^^^^^^^^^^

In PyTorch, tensors can be located in RAM (cpu) or on the GPU (cuda). The
location of tensors is always preserved by tomosipo, i.e., the location of the
output always equals the location of the input:

.. doctest:: operator
   :skipif: not cuda_available or not torch_available

   >>> A(x.cuda()).device
   device(type='cuda', index=0)
   >>> A(x.cpu()).device
   device(type='cpu')

.. warning::

   If the data is located on GPU, then silent data corruption may occur when the
   volume or projection data is not a multiple of 32 pixels/voxels wide. See:
   https://github.com/ahendriksen/tomosipo/issues/6.

Data width coercion
^^^^^^^^^^^^^^^^^^^

Only 32-bit floats can be projected. If a 64-bit float array is provided, it
will automatically be converted and a warning will be issued:

.. doctest:: operator
   :skipif: not cuda_available

   >>> x = np.ones(A.domain_shape, dtype=np.float64)
   >>> y = A(x)

::

   UserWarning: The parameter initial_value is of type float64; expected `np.float32`.
   The type has been Automatically converted.
   Use `ts.link(x.astype(np.float32))' to inhibit this warning.

Note that the conversion makes a copy of the data. This may be unwanted in
memory-constrained situations.

Contiguity coercion
^^^^^^^^^^^^^^^^^^^

The projection operator requires that the input is laid out in memory
contiguously. An array may be discontiguous if it is has been indexed with a
step size, for instance. In this case, an automatic copy will be made to make
the input array contiguous and a warning will be issued.

.. doctest:: operator
   :skipif: not cuda_available

   >>> x = np.ones((32, 32, 64), dtype=np.float32)
   >>> x = x[:, :, ::2]
   >>> x.shape
   (32, 32, 32)
   >>> y = A(x)

::

   UserWarning: The parameter initial_value should be C_CONTIGUOUS and ALIGNED.
   It has been automatically made contiguous and aligned.
   Use `ts.link(np.ascontiguousarray(x)**' to inhibit this warning.


Large datasets and multiple GPUs
--------------------------------

When multiple GPUs are present, the computation of the projection operator can
be distributed over the available GPUs. How this is done depends on the location
of the input array.

**CPU**: If the input array is located in the system RAM, then the computation
by default takes place on a *single* GPU. To use more than one GPU, use the
following snippet:

.. code-block:: python

   import astra
   gpus = [0, 1, 2, 3]  # Use four gpus, indexed 0 to 4
   astra.set_gpu_index(gpus)

After this code has executed, the ASTRA-toolbox will automatically divide all
projection and backprojection over all four GPUs.

**GPU**: If the input array array is located on a GPU, then the output array
will be located on the same GPU. All computations will take place on this GPU as
well. If you want to use multiple GPUs, you must distribute the data over
multiple GPUs yourself.

Back-end Limits
---------------

Tomosipo uses the ASTRA-toolbox to perform the tomographic projection. We
document some of the known limits here:

* Maximum number of angles: 2^14 (16384) https://github.com/astra-toolbox/astra-toolbox/issues/278
* GPU data strides (detector must be 32 floats wide) https://github.com/astra-toolbox/astra-toolbox/issues/280
* Volume: At least 2^20 voxels (in any direction).
* Width of detector (roughly 2^28 pixels (268 million))
* Height of detector (roughly 2^22 pixels (4 million))
