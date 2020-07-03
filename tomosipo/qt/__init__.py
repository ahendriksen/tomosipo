import warnings
try:
    import pyqtgraph
    from pyqtgraph.Qt import QtCore
    import pyqtgraph.opengl
except ModuleNotFoundError:
    warnings.warn(
        "\n------------------------------------------------------------\n\n"
        "Cannot import all required QT packages. \n"
        "Please make sure to install pyqtgraph, pyqt5, and pyopengl. \n"
        "Currently, you can install these using: \n\n"
        " > conda install pyqtgraph pyqt pyopengl \n"
        "\n------------------------------------------------------------\n\n"
    )
    raise


from .display import display
from . import data
from . import geometry
from . import oriented_box
