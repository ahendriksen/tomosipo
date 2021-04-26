import pytest
import tomosipo as ts
import numpy as np
from . import skip_if_no_qt

try:
    from tomosipo.qt import display, animate
except ModuleNotFoundError:
    pass


@pytest.fixture
def geometry_list():
    T = ts.translate((2, 0, 0))
    R = ts.rotate(pos=0, axis=(1, 0, 0), rad=np.linspace(0, np.pi, 20))
    return [
        ts.volume(),
        T * R * ts.volume().to_vec(),
        ts.parallel(angles=10, shape=2),
    ]


@skip_if_no_qt
def test_video_array(geometry_list):
    animation = animate(*geometry_list)
    assert len(animation.video_as_array()) == 20


@skip_if_no_qt
def test_save(geometry_list, tmpdir, interactive):
    animation = animate(*geometry_list)
    path = tmpdir / "test.mp4"
    animation.save(path)
    assert path.exists()
    assert path.stat().size == 78455


@skip_if_no_qt
def test_display(geometry_list, interactive):
    if interactive:
        display(*geometry_list)
