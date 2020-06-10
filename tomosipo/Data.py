import numpy as np
import astra
import tomosipo as ts
import warnings
import pyqtgraph as pq
from tomosipo.display import get_app, run_app


def data(geometry, initial_value=None):
    """Create a managed Astra Data3d object


    :param geometry: `VolumeGeometry` or `ProjectionGeometry`
        A geometry associated with this dataset.
    :param initial_value: `float` or `np.array` or `ts.Data.Data` (optional)
        An initial value for the data. The default is zero. If a
        numpy array is provided, the array is linked to the astra
        toolbox, i.e. they share the same underlying memory.

        If a `Data' object is passed, it is checked to have
    :returns:
        An initialized dataset.
    :rtype: Data

    """
    # If an instance of Data is passed as initial_value, return it
    # instead of creating a new Data instance.
    if isinstance(initial_value, Data):
        if geometry == initial_value.geometry:
            return initial_value
        else:
            raise ValueError(
                f"Got initial_value={initial_value}, "
                f"but its geometry does not match {geometry}."
            )

    return Data(geometry, initial_value)


class Data(object):
    """Data: a data manager for Astra

    """

    def __init__(self, geometry, initial_value=None):
        """Create a managed Astra Data3d object

        :param geometry: `VolumeGeometry` or `ProjectionGeometry`
            A geometry associated with this dataset.
        :param initial_value: `float` or `np.array` (optional)
            An initial value for the data. The default is zero. If a
            numpy array is provided, the array is linked to the astra
            toolbox, i.e. they share the same underlying memory.
        :returns:
            An initialized dataset.
        :rtype: Data

        """
        super(Data, self).__init__()
        self.geometry = geometry
        self.initial_value = initial_value
        self.astra_geom = geometry.to_astra()

        if self.is_volume():
            astra_data_type = "-vol"
            shape = geometry.shape
        elif self.is_projection():
            astra_data_type = "-sino"
            shape = (geometry.det_shape[0], geometry.num_angles, geometry.det_shape[1])
        else:
            raise ValueError(
                f"Geometry '{geometry.__class__}' is not supported. Cannot determine if volume or projection geometry."
            )

        self._backend = NumpyBackend(shape, initial_value)

        self.astra_id = astra.data3d.link(
            astra_data_type, self.astra_geom, self._backend.get_linkable_array()
        )

    def clone(self):
        """Clone Data object

        Creates a new data object with the same (but copied) data and
        new associated astra geometry.

        :returns: a fresh Data object
        :rtype: Data

        """
        data_copy = np.copy(self.data)
        return Data(self.geometry, data_copy)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        astra.data3d.delete(self.astra_id)

    @property
    def data(self):
        """Returns the underlying data.

        Changes to the return value will be reflected in the astra
        data.

        If you want to avoid this, consider copying the data
        immediately, using `np.copy` for instance.

        NOTE: if this data encapsulates projection data, the order of
        the axes is (V, num_angles, U).

        :returns: The underlying data object. This can be a numpy
        array or some other type of data.

        :rtype: np.array

        """
        return self._backend.data

    @data.setter
    def data(self, val):
        self._backend.data = val

    def is_volume(self):
        return ts.geometry.is_volume(self.geometry)

    def is_projection(self):
        return ts.geometry.is_projection(self.geometry)

    def to_astra(self):
        """Returns astra data id associated with current object

        :returns:
        :rtype:

        """
        return self.astra_id



class NumpyBackend(object):
    """Documentation for NumpyBackend

    """
    def __init__(self, shape, initial_value):
        super(NumpyBackend, self).__init__()
        self._shape = shape

        if initial_value is None:
            self._data = np.zeros(shape, dtype=np.float32)
        elif np.isscalar(initial_value):
            self._data = np.zeros(shape, dtype=np.float32)
            self._data[:] = initial_value
        else:
            initial_value = np.array(initial_value, copy=False)
            # Make contiguous:
            if initial_value.dtype != np.float32:
                warnings.warn(
                    f"The parameter initial_value is of type {initial_value.dtype}; expected `np.float32`. "
                    f"The type has been Automatically converted."
                )
                initial_value = initial_value.astype(np.float32)
            if not (
                initial_value.flags["C_CONTIGUOUS"] and initial_value.flags["ALIGNED"]
            ):
                warnings.warn(
                    f"The parameter initial_value should be C_CONTIGUOUS and ALIGNED."
                    f"It has been automatically made contiguous and aligned."
                )
                initial_value = np.ascontiguousarray(initial_value)
            self._data = initial_value

    def get_linkable_array(self):
        return self._data

    @property
    def data(self):
        """Returns a shared numpy array with the underlying data.

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
        if val != self._data:
            raise AttributeError(
                "You cannot change which numpy array backs a dataset.\n"
                "To change the underlying data instead, use: \n"
                " >>> vd.data[:] = new_data\n"
            )


@ts.display.register(Data)
def display_data(d):
    """Display a projection or volume data set.

    Shows the slices or projection images depending on the argument.

    For projection datasets, the "first" pixel (0, 0) is located
    in the lower-left corner and the "last" pixel (N, N) is located in
    the top-right corner.

    For volume datasets, the voxel (0, 0, 0) is located in the
    lower-left corner of the first (left-most) slice and the voxel (N,
    N, N) is located in the top-right corner of the last slice.

    :param d: `Data`
        A tomosipo dataset of either a volume or projection set.
    :returns: None
    :rtype:

    """

    if d.is_volume():
        app = get_app()
        pq.image(d.data, scale=(1, -1))
        run_app(app)
    elif d.is_projection():
        app = get_app()
        pq.image(d.data, scale=(1, -1), axes=dict(zip("ytx", range(3))))
        run_app(app)
