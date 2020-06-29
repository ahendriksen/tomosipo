"""This module adds support for torch arrays as astra.data3d backends

This module is not automatically imported by tomosipo, you must import
it manually as follows:

>>> import tomosipo.torch_support

Now, you may use torch tensors as you would numpy arrays:

>>> vg = ts.volume(shape=(10, 10, 10))
>>> vd = ts.data(vg, torch.zeros(10, 10, 10))
>>> # Or directly on gpu:
>>> vd = ts.data(vg, torch.zeros(10, 10, 10).cuda())

NOTE: Support for data on multiple and/or non-default GPUs has not yet
been tested. It is unsupported(!).

"""
import astra
from tomosipo.Data import backends
import warnings
import torch


class TorchBackend(object):
    """Documentation for TorchBackend

    """
    def __init__(self, shape, initial_value):
        super(TorchBackend, self).__init__()
        self._shape = shape

        if not isinstance(initial_value, torch.Tensor):
            raise ValueError(f"Expected initial_value to be a `torch.Tensor'. Got {initial_value.__class__}")

        if initial_value.shape == torch.Size([]):
            self._data = torch.zeros(shape, dtype=torch.float32, device=initial_value.device)
            self._data[:] = initial_value
        else:
            if shape != initial_value.shape:
                raise ValueError(f"Expected initial_value with shape {shape}. Got {initial_value.shape}")
            # Ensure float32
            if initial_value.dtype != torch.float32:
                warnings.warn(
                    f"The parameter initial_value is of type {initial_value.dtype}; expected `torch.float32`. "
                    f"The type has been Automatically converted."
                )
                initial_value = initial_value.to(dtype=torch.float32)
            # Make contiguous:
            if not initial_value.is_contiguous():
                warnings.warn(
                    f"The parameter initial_value should be contiguous. "
                    f"It has been automatically made contiguous."
                )
                initial_value = initial_value.contiguous()
            self._data = initial_value

    @staticmethod
    def accepts_initial_value(initial_value):
        # only accept torch tensors
        if isinstance(initial_value, torch.Tensor):
            return True
        else:
            return False

    def get_linkable_array(self):
        if self._data.is_cuda:
            z, y, x = self._data.shape
            pitch = x * 4       # we assume 4 byte float32 values
            link = astra.data3d.GPULink(self._data.data_ptr(), x, y, z, pitch)
            return link
        else:
            # XXX: We assume a tensor is either cuda or on cpu..

            # The torch tensor may be part of the computation
            # graph. It must be detached to obtain a numpy
            # array. We assume that this function will only be
            # called to feed the data into Astra, which should not
            # modify it. So this should be fine.
            return self._data.detach().numpy()

    def new_zeros(self, shape):
        return TorchBackend(
            shape,
            self._data.new_zeros(shape)
        )

    def new_full(self, shape, value):
        return TorchBackend(
            shape,
            self._data.new_full(shape, value)
        )

    @property
    def data(self):
        """Returns a shared array with the underlying data.

        Changes to the return value will be reflected in the astra
        data.

        If you want to avoid this, consider copying the data
        immediately, using `np.copy` for instance.

        NOTE: if the underlying object is an Astra projection data
        type, the order of the axes will be in (Y, num_angles, X)
        order.

        :returns: np.array
        :rtype: np.array

        """
        return self._data

    @data.setter
    def data(self, val):
        raise AttributeError(
            "You cannot change which torch tensor backs a dataset.\n"
            "To change the underlying data instead, use: \n"
            " >>> vd.data[:] = new_data\n"
        )


backends.append(TorchBackend)
