import tomosipo as ts
import numpy as np


def test_display_parallel_geometry(interactive):
    if interactive:
        from tomosipo.qt import display
    pg = ts.parallel(angles=100, shape=100)
    vg = ts.volume()
    if interactive:
        display(vg, pg)

    # Test with multiple geometries:
    pg2 = ts.parallel(angles=10, size=2.0)
    vg2 = ts.volume().translate((1.0, 0, 0))
    if interactive:
        display(vg, vg2, pg, pg2)


def test_display_cone_geometry(interactive):
    if interactive:
        from tomosipo.qt import display

    pg = ts.cone(angles=100, src_obj_dist=10, src_det_dist=15)
    vg = ts.volume()

    # Add volume vector geometry
    R = ts.rotate(pos=0, axis=(1, 0, 0), rad=-np.linspace(0, np.pi, 300))
    T = ts.translate((2, 0, 0))
    vg_vec = T * R * vg.to_vec()

    if interactive:
        display(pg, vg, vg_vec)

    # Test with two projection geometries:
    pg2 = ts.cone(angles=50, src_obj_dist=5, src_det_dist=7.5)
    if interactive:
        display(vg, vg_vec, pg, pg2)
