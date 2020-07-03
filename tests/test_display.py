import tomosipo as ts


def test_display_data(interactive):
    if interactive:
        from tomosipo.qt import display

    p = ts.data(ts.cone(angles=100, shape=100))
    v = ts.data(ts.volume_from_projection_geometry(p.geometry).reshape(100))

    # Fill v with hollow box phantom
    ts.phantom.hollow_box(v)
    ts.forward(v, p)

    if interactive:
        display(v)
    if interactive:
        display(p)
