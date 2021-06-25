.. _intro_fast_reconstruction:

Fast SIRT reconstruction using PyTorch on the GPU
=================================================

.. note::
   To keep the size of the documentation in version control manageable, we do
   not include images of the volume and the reconstruction. It is recommended to
   follow along in your favorite python environment so you can see what is going
   on.

In the :ref:`previous tutorial <intro_simple_reconstruction>`, we implemented
SIRT reconstruction using NumPy. This was not very fast, because intermediate
data had to be moved back and forth to the GPU during the execution of the
algorithm. In this tutorial, we implement SIRT using PyTorch, which can compute
on both the CPU and the GPU. Before you continue, make sure to :ref:`install
pytorch <intro_install_with_pytorch>`.

To get a better feel for the difference in speed, we use a larger volume and
detector:

.. testcode:: session

   import tomosipo as ts
   import numpy as np

   N = 256
   M = (N * 3) // 2
   vg = ts.volume(shape=(N, N, N))
   pg = ts.parallel(angles=M, shape=(M, M))
   A = ts.operator(vg, pg)


Torch support is automatic if it has been installed in the host environment. To
use torch, we import it:

.. testcode:: session
   :skipif: (not cuda_available) or (not torch_available)

   import torch


Using torch, we recreate the hollow cube phantom and forward project to obtain
the sinogram stack `y`:

.. testcode:: session
   :skipif: (not cuda_available) or (not torch_available)

   x = torch.ones(A.domain_shape)
   x[8:-8, 8:-8, 8:-8] = 0.0
   y = A(x)

Note that torch uses `float32` values by default, so we do not have to specify
`dtype=torch.float32` explicitly. Also, `y` is a torch tensor, because the input
to the operator `A` is a torch tensor:

.. doctest:: session
   :skipif: (not cuda_available) or (not torch_available)

   >>> y.dtype
   torch.float32

Now we prepare the preconditioning matrices `R` and `C` using torch:

.. testcode:: session
   :skipif: (not cuda_available) or (not torch_available)

   R = 1 / A(torch.ones(A.domain_shape))
   torch.clamp(R, max=1 / ts.epsilon, out=R)
   C = 1 / A.T(torch.ones(A.range_shape))
   torch.clamp(C, max=1 / ts.epsilon, out=C)

Next, we reconstruct from the sinogram stack `y` into `x_rec`:

.. testcode:: session
   :skipif: True

   num_iterations = 50
   x_rec = torch.zeros(A.domain_shape)

   for i in range(num_iterations):
       x_rec += C * A.T(R * (y - A(x_rec)))

This code is in fact not much faster than the NumPy code from the previous
tutorial. We still use tensors that are stored "on the CPU", i.e., system RAM.
We can create a reconstruction function that works on tensors whose `device`
location is either the CPU or GPU:

.. testcode:: session
   :skipif: (not cuda_available) or (not torch_available)

   def sirt(A, y, num_iterations=10):
       dev = y.device
       R = 1 / A(torch.ones(A.domain_shape, device=dev))
       torch.clamp(R, max=1 / ts.epsilon, out=R)
       C = 1 / A.T(torch.ones(A.range_shape, device=dev))
       torch.clamp(C, max=1 / ts.epsilon, out=C)

       x_rec = torch.zeros(A.domain_shape, device=dev)

       for i in range(num_iterations):
           x_rec += C * A.T(R * (y - A(x_rec)))

       return x_rec


.. testcode:: session
   :skipif: True

   from timeit import default_timer as timer
   y_cpu = y
   y_gpu = y.to("cuda")

   start_cpu = timer()
   sirt(A, y_cpu)
   end_cpu = timer()
   start_gpu = timer()
   sirt(A, y_gpu)
   end_gpu = timer()

   print(f"cpu : {end_cpu - start_cpu:0.2f} seconds")
   print(f"gpu : {end_gpu - start_gpu:0.2f} seconds")


::

   cpu : 4.95 seconds
   gpu : 2.46 seconds

As you can see, the GPU code is almost twice as fast! 

The SIRT algorithm is implemented in the `ts_algorithms
<https://github.com/ahendriksen/ts_algorithms>`_ package with some additional
optimizations. This package contains some well-tested reconstruction algorithms
for use with `tomosipo`.

