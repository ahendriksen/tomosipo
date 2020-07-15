import numpy as np
import tomosipo as ts
from . import (
    Transform,
    ConeGeometry,
    ConeVectorGeometry,
    ParallelGeometry,
    ParallelVectorGeometry,
    VolumeGeometry,
    OrientedBox,
)


def concatenate(items):
    """Concatenate geometries and transformations

    Supports:
    - parallel geometries (vec and non-vec)
    - cone geometries (vec and non-vec)
    - oriented boxes
    - transformations

    Parallel and Cone geometries are converted to vector geometries.

    :returns:
    :rtype:

    """
    if len(items) == 0:
        raise ValueError("ts.concatenate expected at least one argument. ")

    def is_cone(x):
        return isinstance(x, ConeGeometry) or isinstance(x, ConeVectorGeometry)

    def is_parallel(x):
        return isinstance(x, ParallelGeometry) or isinstance(x, ParallelVectorGeometry)

    if all(isinstance(i, Transform) for i in items):
        return Transform(np.concatenate([i.matrix for i in items]))

    if all(is_parallel(i) for i in items):
        if not all(i.det_shape == items[0].det_shape for i in items):
            raise ValueError(
                "Cannot concatenate geometries. Not all detector shapes are equal."
            )
        return ParallelVectorGeometry(
            items[0].det_shape,
            ray_dir=np.concatenate([i.ray_dir for i in items]),
            det_pos=np.concatenate([i.det_pos for i in items]),
            det_v=np.concatenate([i.det_v for i in items]),
            det_u=np.concatenate([i.det_u for i in items]),
        )
    if all(is_cone(i) for i in items):
        if not all(i.det_shape == items[0].det_shape for i in items):
            raise ValueError(
                "Cannot concatenate geometries. Not all detector shapes are equal."
            )
        return ConeVectorGeometry(
            items[0].det_shape,
            src_pos=np.concatenate([i.src_pos for i in items]),
            det_pos=np.concatenate([i.det_pos for i in items]),
            det_v=np.concatenate([i.det_v for i in items]),
            det_u=np.concatenate([i.det_u for i in items]),
        )
    if all(isinstance(i, OrientedBox) for i in items):
        if not all(i.rel_size == items[0].rel_size for i in items):
            raise ValueError(
                "Cannot concatenate oriented boxes. Not all sizes are equal."
            )

        return OrientedBox(
            size=items[0].rel_size,
            pos=np.concatenate([i.pos for i in items]),
            w=np.concatenate([i.w for i in items]),
            v=np.concatenate([i.v for i in items]),
            u=np.concatenate([i.u for i in items]),
        )

    types = set(type(i) for i in items)
    raise TypeError(f"Concatenating objects of types {types} is not supported. ")
