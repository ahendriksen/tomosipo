Share an interactive notebook on Google Colab
=============================================

If you want to use the ASTRA-toolbox and tomosipo in an environment such as
`Google Colab <https://colab.research.google.com/>`_, then you can install
tomosipo as follows:

.. code-block:: python

   ! apt install build-essential autoconf libtool
   ! pip install cython
   ! git clone https://github.com/astra-toolbox/astra-toolbox.git
   ! cd astra-toolbox/build/linux && ./autogen.sh && ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=module
   ! cd astra-toolbox/build/linux && make install -j 4
   ! pip install git+https://github.com/ahendriksen/tomosipo.git

An example can be found `here
<https://colab.research.google.com/github/ahendriksen/tomosipo/blob/update-readme/notebooks/00_getting_started_google_colab.ipynb>`_.
