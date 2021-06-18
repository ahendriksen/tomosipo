Execute the unit tests
======================


Run the tests
-------------

To run all tests, execute:

.. code-block:: bash

    pytest tests/

To skip slow tests, execute:

.. code-block:: bash

    pytest -m "not slow" tests/

To only run only the tests that failed in the previous run, execute:

.. code-block:: bash

    pytest tests/ --lf
