========================
Update the documentation
========================

Update API reference
--------------------

To update the API reference, execute:

.. code-block:: bash

    sphinx-apidoc -M -f -e --tocfile api_reference -H "API Reference" --ext-autodoc  -o doc/ref/ tomosipo

This is only needed when the module hierarchy has changed.

Generate documentation
----------------------

To generate the documentation in the directory `public`, execute:

.. code-block:: bash

    sphinx-build -b html doc/ public


Run code in documentation
-------------------------

To run the code that is embedded in the running text of the documentation, execute:

.. code-block:: bash

    python -msphinx -b doctest doc/ ./.doctest-output

The status of the tests will be written to `.doctest-output`.
