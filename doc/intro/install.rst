Install tomosipo
================

A minimal installation requires:

* python >= 3.6
* ASTRA-toolbox (the latest 1.9.x development version is *required*)
* CUDA

Installation using anaconda
---------------------------

The requirements can be installed using the anaconda package manager. The
following snippet creates a new conda environment named `tomosipo` (replace
`X.X` by your CUDA version)

.. code-block:: bash

   conda create -n tomosipo cudatoolkit=<X.X> tomosipo -c defaults -c astra-toolbox/label/dev -c aahendriksen


Install the latest development branch
-------------------------------------

To install the latest development branch from GitHub, first create a new
environment named `tomosipo` containing the required packages:

.. code-block:: bash

    conda create -n tomosipo python>=3.6 astra-toolbox cudatoolkit=X.X -c astra-toolbox/label/dev

Then activate the environment and install tomosipo using pip:

.. code-block:: bash

    source activate tomosipo
    pip install git+https://github.com/ahendriksen/tomosipo@develop

Install optional dependencies
-----------------------------

To use tomosipo with PyTorch, QT, ODL, and cupy, install:

.. code-block:: bash

    conda create -n tomosipo tomosipo cudatoolkit=<X.X> pytorch cupy pyqtgraph pyqt pyopengl cupy \
                 -c defaults -c astra-toolbox/label/dev -c pytorch -c conda-forge -c aahendriksen
    source activate tomosipo
    # Install latest version of ODL:
    pip install git+https://github.com/odlgroup/odl

.. _intro_install_with_pytorch:

Install with pytorch
--------------------

To just install PyTorch, use

.. code-block:: bash

   conda create -n tomosipo pytorch cudatoolkit=<X.X> tomosipo -c defaults -c astra-toolbox/label/dev -c aahendriksen -c pytorch
