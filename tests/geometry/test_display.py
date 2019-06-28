import tomosipo as ts

interactive = True


def test_display_parallel_geometry():
    pg = ts.parallel(angles=100, shape=100)
    vg = ts.volume()
    if interactive:
        ts.display(pg, vg)


def test_display_random_parallel_geometry():
    pg = ts.geometry.random_transform()(ts.parallel(angles=100, shape=100).to_vec())
    vg = ts.volume()
    if interactive:
        ts.display(pg, vg)


def test_display_cone_geometry():
    pg = ts.cone(angles=100, src_obj_dist=10, src_det_dist=15)
    vg = ts.volume_from_projection_geometry(pg, inside=False)

    if interactive:
        ts.display(pg, vg)
