###############################################################################
#                               Check pyqtgraph                               #
###############################################################################
def __is_pyqtgraph_available():
    try:
        import pyqtgraph
        from pyqtgraph.Qt import QtCore
        import pyqtgraph.opengl
    except ModuleNotFoundError:
        return False

    return True


if not __is_pyqtgraph_available():
    raise ModuleNotFoundError(
        "\n------------------------------------------------------------\n\n"
        "Cannot import all required QT packages. \n"
        "Please make sure to install pyqtgraph, pyqt5, and pyopengl. \n"
        "You can install these using: \n\n"
        " > conda install pyqtgraph pyqt pyopengl \n"
        "\n------------------------------------------------------------\n\n"
    )


###############################################################################
#                                 Check ffmpeg                                #
###############################################################################
def __is_ffmpeg_available():
    # check for ffmpeg binaries:
    import shutil

    if shutil.which("ffmpeg") is None:
        return False

    try:
        import ffmpeg
    except ModuleNotFoundError:
        return False

    return True


if not __is_ffmpeg_available():
    raise ModuleNotFoundError(
        "\n------------------------------------------------------------\n\n"
        "FFMPEG encoding is not available. Make sure that ffmpeg is installed using: \n"
        "> conda install ffmpeg ffmpeg-python -c defaults -c conda-forge \n\n"
        "If using jupyterlab, you might have to install ffmpeg in the base\n"
        "environment as well. \n"
        "\n------------------------------------------------------------\n\n"
    )

###############################################################################
#                                  Submodules                                 #
###############################################################################
from .display import display
from .animation import animate
from . import data
from . import geometry
