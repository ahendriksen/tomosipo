import tomosipo as ts


def from_astra(astra_geom):
    """Import ASTRA geometry to tomosipo

    :param astra_geom: an ASTRA 3D volume or projection geometry
    :returns: a tomosipo 3D volume or projection geometry
    :rtype:

    """
    if not isinstance(astra_geom, dict):
        raise TypeError(
            f"Currently, tomosipo only supports importing ASTRA geometries. "
            f"Objects of type {type(astra_geom)} are not supported. "
            f"Perhaps you meant to use `ts.to_astra'? "
        )
    if "GridSliceCount" in astra_geom:
        return ts.geometry.volume.from_astra(astra_geom)
    else:
        return ts.geometry.conversion.from_astra_projection_geometry(astra_geom)


def to_astra(x):
    try:
        return x.to_astra()
    except AttributeError:
        raise TypeError(
            f"The object of type {type(x)} does not support conversion to ASTRA."
            f"Perhaps you meant to use `ts.from_astra'? "
        )
