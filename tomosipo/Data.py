import astra
import tomosipo as ts


def data(geometry, initial_value=None):
    """Create a managed Astra Data3d object

    :param geometry: `VolumeGeometry` or `ProjectionGeometry`
        A geometry associated with this dataset.
    :param initial_value: `float` or `np.array`
        An initial value for the data. The default is zero. If a
        numpy array is provided, the array is linked to the astra
        toolbox, i.e. they share the same underlying memory.
    :returns: An initialized dataset.
    :rtype: `Data`

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
    """Data: a data manager for Astra"""

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
        :rtype: `Data`

        """
        super(Data, self).__init__()
        self.geometry = geometry
        if not hasattr(geometry, "to_astra"):
            raise TypeError(
                f"Cannot create data object with geometry because it is not convertible to ASTRA: {geometry}"
            )
        self.astra_geom = geometry.to_astra()

        if self.is_volume():
            astra_data_type = "-vol"
        elif self.is_projection():
            astra_data_type = "-sino"
        else:
            raise ValueError(
                f"Geometry '{type(geometry)}' is not supported. Cannot determine if volume or projection geometry."
            )

        self._link = ts.link(geometry, initial_value)

        self.astra_id = astra.data3d.link(
            astra_data_type, self.astra_geom, self._link.linked_data
        )

    def clone(self):
        """Clone Data object

        Creates a new data object with the same (but copied) data and
        new associated astra geometry.

        :returns: a fresh Data object
        :rtype: Data

        """
        data_copy = self._link.clone().data
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

        :returns:
            The underlying data object. This can be a numpy array or
            some other type of data.
        :rtype: `np.array`

        """
        return self._link.data

    @data.setter
    def data(self, val):
        self._link.data = val

    @property
    def link(self):
        return self._link

    # TODO: Implement .numpy() so that display_data can also show data from gpu..

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
