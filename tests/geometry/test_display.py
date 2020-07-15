import tomosipo as ts


def test_display_parallel_geometry(interactive):
    if interactive:
        from tomosipo.qt import display
    pg = ts.parallel(angles=100, shape=100)
    vg = ts.volume()
    if interactive:
        display(vg, pg)

    # Test with multiple geometries:
    pg2 = ts.parallel(angles=10, size=2.0)
    vg2 = ts.volume().translate(1.0)
    if interactive:
        display(vg, vg2, pg, pg2)


def test_display_cone_geometry(interactive):
    if interactive:
        from tomosipo.qt import display

    pg = ts.cone(angles=100, src_obj_dist=10, src_det_dist=15)
    vg = ts.volume()

    if interactive:
        display(pg, vg)

    # Test with two projection geometries:
    pg2 = ts.cone(angles=50, src_obj_dist=5, src_det_dist=7.5)
    if interactive:
        display(vg, pg, pg2)
