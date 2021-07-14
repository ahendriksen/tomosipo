import tomosipo as ts
import numpy as np
from . import skip_if_no_cuda
from tomosipo.astra import forward


@skip_if_no_cuda
def test_display_data(interactive):
    if interactive:
        from tomosipo.qt import display

    p = ts.data(ts.cone(size=np.sqrt(2), cone_angle=1 / 2, angles=100, shape=100))
    v = ts.data(ts.volume(shape=100))

    # Fill v with hollow box phantom
    ts.phantom.hollow_box(v)
    forward(v, p)

    if interactive:
        display(v)
    if interactive:
        display(p)
