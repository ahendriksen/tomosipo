from .base_projection import ProjectionGeometry
from .cone import ConeGeometry
from .cone_vec import ConeVectorGeometry

# TODO: change the name of this module.

# The name of this module is unfortunately chosen. The
# from_astra_geometry function cannot be defined in
# ProjectionGeometry.py: that would cause a circular import.


def from_astra_geometry(astra_pg):
    pg_type = astra_pg["type"]
    if pg_type == "cone":
        return ConeGeometry.from_astra(astra_pg)
    elif pg_type == "cone_vec":
        return ConeVectorGeometry.from_astra(astra_pg)
    elif pg_type == "parallel3d_vec":
        raise NotImplementedError(
            "ProjectionGeometry.from_astra does not yet support parallel3d_vec geometries."
        )
    elif pg_type == "parallel3d":
        raise NotImplementedError(
            "ProjectionGeometry.from_astra does not yet support parallel3d geometries."
        )
    else:
        raise ValueError(
            "ProjectionGeometry.from_astra only supports 3d astra geometries"
        )
