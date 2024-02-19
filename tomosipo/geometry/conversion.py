from .base_projection import ProjectionGeometry
from .cone import ConeGeometry
from .cone_vec import ConeVectorGeometry
from .cyl_cone_vec import CylConeVectorGeometry
from .det_vec import DetectorVectorGeometry
from .parallel_vec import ParallelVectorGeometry
from .parallel import ParallelGeometry

# TODO: change the name of this module.

# The name of this module is unfortunately chosen. The
# from_astra_projection_geometry function cannot be defined in
# ProjectionGeometry.py: that would cause a circular import.


def from_astra_projection_geometry(astra_pg):
    pg_type = astra_pg["type"]
    if pg_type == "cone":
        return ConeGeometry.from_astra(astra_pg)
    elif pg_type == "cone_vec":
        return ConeVectorGeometry.from_astra(astra_pg)
    elif pg_type == "cyl_cone_vec":
        return CylConeVectorGeometry.from_astra(astra_pg)
    elif pg_type == "det_vec":
        return DetectorVectorGeometry.from_astra(astra_pg)
    elif pg_type == "parallel3d_vec":
        return ParallelVectorGeometry.from_astra(astra_pg)
    elif pg_type == "parallel3d":
        return ParallelGeometry.from_astra(astra_pg)
    else:
        raise ValueError(
            "ProjectionGeometry.from_astra only supports 3d astra geometries"
        )
