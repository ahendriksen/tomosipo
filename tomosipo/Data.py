import numpy as np
import astra
import tomosipo as ts
import warnings


def data(geometry, initial_value=None):
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

        astra_type_dict = {
            "vol": "-vol",
            "vol_link": "-vol",
            "sino": "-proj3d",
            "sino_link": "-sino",
        }

        if self.is_volume():
            self.astra_geom = geometry.to_astra()
            self.data_type = "vol"
        elif self.is_projection():
            self.astra_geom = geometry.to_astra()
            self.data_type = "sino"
        else:
            raise ValueError(
                f"Geometry '{geometry.__class__}' is not supported. Cannot determine if volume or projection geometry."
            )

        if initial_value is None:
            astra_type = astra_type_dict[self.data_type]
            self.astra_id = astra.data3d.create(astra_type, self.astra_geom, 0.0)
        elif np.isscalar(initial_value):
            astra_type = astra_type_dict[self.data_type]
            self.astra_id = astra.data3d.create(astra_type, self.astra_geom, initial_value)
        else:
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
            astra_type = astra_type_dict[self.data_type + "_link"]
            self.astra_id = astra.data3d.link(astra_type, self.astra_geom, initial_value)

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
        return astra.data3d.get_shared(self.astra_id)

    @data.setter
    def data(self, val):
        raise ValueError(
            "You cannot change which numpy array backs a dataset.\n"
            "To change the underlying data instead, use: \n"
            " >>> vd.data[:] = new_data\n"
        )

    def is_volume(self):
        return ts.is_volume_geometry(self.geometry)

    def is_projection(self):
        return ts.ProjectionGeometry.is_projection_geometry(self.geometry)

    def to_astra(self):
        """Returns astra data id associated with current object

        :returns:
        :rtype:

        """
        return self.astra_id
