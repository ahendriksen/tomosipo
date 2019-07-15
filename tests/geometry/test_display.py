import tomosipo as ts


def test_display_parallel_geometry(interactive):
    pg = ts.parallel(angles=100, shape=100)
    vg = ts.volume()
    if interactive:
        ts.display(vg, pg)

    # Test with multiple geometries:
    pg2 = ts.parallel(angles=10, size=2.0)
    vg2 = ts.volume().translate(1.0)
    if interactive:
        ts.display(vg, vg2, pg, pg2)


def test_display_cone_geometry(interactive):
    pg = ts.cone(angles=100, src_obj_dist=10, src_det_dist=15)
    vg = ts.volume_from_projection_geometry(pg, inside=False)

    if interactive:
        ts.display(pg, vg)

    # Test with two projection geometries:
    pg2 = ts.cone(angles=50, src_obj_dist=5, src_det_dist=7.5)
    if interactive:
        ts.display(vg, pg, pg2)
