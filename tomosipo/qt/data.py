from tomosipo.Data import Data
import pyqtgraph as pq
from .display import (
    run_app,
    get_app,
    display_backends,
)


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


display_backends[Data] = display_data
