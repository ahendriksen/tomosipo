import pytest

try:
    import tomosipo.qt

    qt_present = True
except ModuleNotFoundError:
    qt_present = False

skip_if_no_qt = pytest.mark.skipif(not qt_present, reason="QT not installed")
