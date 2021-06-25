Build the conda package
=======================

To build the anaconda package of the project, execute:

.. code-block:: bash

    conda install conda-build anaconda-client
    conda build conda/ -c astra-toolbox/label/dev

