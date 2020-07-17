import numpy as np
import tomosipo as ts
from . import (
    Transform,
    ConeGeometry,
    ConeVectorGeometry,
    ParallelGeometry,
    ParallelVectorGeometry,
    VolumeGeometry,
    VolumeVectorGeometry,
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

    def is_volume(x):
        return isinstance(x, VolumeGeometry) or isinstance(x, VolumeVectorGeometry)

    if all(isinstance(i, Transform) for i in items):
        return Transform(np.concatenate([i.matrix for i in items]))

    if all(is_parallel(i) for i in items):
        if not all(i.det_shape == items[0].det_shape for i in items):
            raise ValueError(
                "Cannot concatenate geometries. Not all detector shapes are equal."
            )
        return ts.parallel_vec(
            shape=items[0].det_shape,
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

    if all(is_volume(i) for i in items):
        if not all(i.shape == items[0].shape for i in items):
            raise ValueError("Cannot concatenate volumes. Not all shapes are equal.")

        return ts.volume_vec(
            shape=items[0].shape,
            pos=np.concatenate([i.to_vec().pos for i in items]),
            w=np.concatenate([i.to_vec().w for i in items]),
            v=np.concatenate([i.to_vec().v for i in items]),
            u=np.concatenate([i.to_vec().u for i in items]),
        )

    types = set(type(i) for i in items)
    raise TypeError(f"Concatenating objects of types {types} is not supported. ")
