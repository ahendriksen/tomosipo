import tomosipo as ts

interactive = False


def test_display_data():
    p = ts.data(ts.cone(angles=100, shape=100))
    v = ts.data(ts.volume_from_projection_geometry(p.geometry).reshape(100))

    # Fill v with hollow box phantom
    ts.phantom.hollow_box(v)
    ts.forward(v, p)

    if interactive:
        ts.display(v)
    if interactive:
        ts.display(p)
