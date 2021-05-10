"""This module adds support for cupy arrays as astra.data3d backends

This module is not automatically imported by tomosipo, you must import
it manually as follows:

>>> import tomosipo as ts
>>> import tomosipo.cupy

Now, you may use cupy arrays as you would numpy arrays:

>>> vg = ts.volume(shape=(10, 10, 10))
>>> pg = ts.parallel(angles=10, shape=10)
>>> A = ts.operator(vg, pg)
>>> x = cupy.zeros(A.domain_shape, dtype='float32')
>>> A(x).shape == A.range_shape
True
"""
import astra
from .base import Link, backends
from contextlib import contextmanager
import warnings
import cupy


class CupyLink(Link):
    """Link implementation for cupy arrays"""

    def __init__(self, shape, initial_value):
        super().__init__(shape, initial_value)

        if not isinstance(initial_value, cupy.ndarray):
            raise ValueError(
                f"Expected initial_value to be a `cupy.ndarray'. Got {initial_value.__class__}"
            )

        if shape != initial_value.shape:
            raise ValueError(
                f"Expected initial_value with shape {shape}. Got {initial_value.shape}"
            )
        # Ensure float32
        if initial_value.dtype != cupy.dtype("float32"):
            warnings.warn(
                f"The parameter initial_value is of type {initial_value.dtype}; expected `cupy.dtype('float32')`. "
                f"The type has been automatically converted. "
                f"Use `ts.link(x.astype('float32'))' to inhibit this warning. "
            )
            initial_value = initial_value.astype("float32")
            # Make contiguous:
            if not (initial_value.flags["C_CONTIGUOUS"]):
                warnings.warn(
                    f"The parameter initial_value should be C_CONTIGUOUS. "
                    f"It has been automatically made contiguous. "
                    f"Use `ts.link(cupy.ascontiguousarray(x))' to inhibit this warning. "
                )
                initial_value = cupy.ascontiguousarray(initial_value)

        self._data = initial_value

    ###########################################################################
    #                      "Protocol" functions / methods                     #
    ###########################################################################
    @staticmethod
    def __accepts__(initial_value):
        # only accept cupy arrays
        return isinstance(initial_value, cupy.ndarray)

    def __compatible_with__(self, other):
        dev_self = self._data.device
        # TODO: Implement compatibility with torch tensors on GPU.
        if isinstance(other, CupyLink):
            dev_other = other._data.device
        else:
            return NotImplemented

        return dev_self == dev_other

    ###########################################################################
    #                                Properties                               #
    ###########################################################################
    @property
    def linked_data(self):
        z, y, x = self._data.shape
        pitch = x * 4  # we assume 4 byte float32 values
        link = astra.data3d.GPULink(self._data.data.ptr, x, y, z, pitch)
        return link

    @property
    def data(self):
        """Returns a shared array with the underlying data.

        Changes to the return value will be reflected in the astra
        data.

        If you want to avoid this, consider copying the data
        immediately, using `x.data.copy()` for instance.

        NOTE: if the underlying object is an Astra projection data
        type, the order of the axes will be in (Y, num_angles, X)
        order.

        :returns: cupy.ndarray
        :rtype: cupy.ndarray

        """
        return self._data

    @data.setter
    def data(self, val):
        raise AttributeError(
            "You cannot change which cupy array backs a dataset.\n"
            "To change the underlying data instead, use: \n"
            " >>> vd.data[:] = new_data\n"
        )

    ###########################################################################
    #                             Context manager                             #
    ###########################################################################
    @contextmanager
    def context(self):
        """Context-manager to manage ASTRA interactions

        This context-manager makes sure that the current CUDA
        stream is set to the CUDA device of the current linked data.

        :returns:
        :rtype:

        """
        with self._data.device:
            yield

    ###########################################################################
    #                            New data creation                            #
    ###########################################################################
    def new_zeros(self, shape):
        # Ensure data is created on same device
        with self.context():
            return CupyLink(shape, cupy.zeros(shape, dtype=self._data.dtype))

    def new_full(self, shape, value):
        # Ensure data is created on same device
        with self.context():
            return CupyLink(shape, cupy.full(shape, value, dtype=self._data.dtype))

    def new_empty(self, shape):
        # Ensure data is created on same device
        with self.context():
            return CupyLink(shape, cupy.empty(shape, dtype=self._data.dtype))

    def clone(self):
        # Ensure data is created on same device
        with self.context():
            return CupyLink(self._data.shape, self._data.copy())


backends.append(CupyLink)
