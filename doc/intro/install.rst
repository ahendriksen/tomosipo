Install tomosipo
================

A minimal installation requires:

- python >= 3.6
- ASTRA-toolbox (the latest 1.9.x development version is *required*)
- CUDA

These requirements can be installed using conda (replace `X.X` by your
CUDA version)

.. code-block:: bash

    conda create -y -n tomosipo python=3.6 astra-toolbox cudatoolkit=X.X -c astra-toolbox/label/dev
    pip install git+https://github.com/ahendriksen/tomosipo@develop
    source activate tomosipo


To use tomosipo with PyTorch, QT, ODL, and cupy, install:

.. code-block:: bash

    conda create -y -n tomosipo python=3.6 astra-toolbox cudatoolkit=<X.X> pytorch cupy pyqtgraph pyqt pyopengl cupy \
                 -c defaults -c astra-toolbox/label/dev -c pytorch -c conda-forge
    source activate tomosipo
    # Install latest version of ODL:
    pip install git+https://github.com/odlgroup/odl
    # Install development version of tomosipo:
    pip install git+https://github.com/ahendriksen/tomosipo@develop
