import pytest
import astra
import doctest

cuda_available = astra.use_cuda()
skip_if_no_cuda = pytest.mark.skipif(not cuda_available, reason="Cuda not available")


def add_doctest_cases(testcase_class, module_to_document):
    """Adds doctests to pytest run

    This function enables you to explicitly add doctests from a certain module
    to your pytest run. This avoids running into trouble with the
    `--doctest-modules` option of pytest. By piggybacking on the unittest
    support of pytest and the unittest support of doctest, we are able to import
    the doctests into the pytest framework.


    Use as follows in a test file:

    .. source-block:: python

        import pytest
        import unittest
        import module_to_test
        from . import add_doctest_cases


        class TestDocs(unittest.TestCase):
            pass


        add_doctest_cases(TestDocs, module_to_test)

    """
    for case in doctest.DocTestSuite(module_to_document):
        case_name = case._dt_test.name.replace('.', '_')
        setattr(testcase_class, f"test_{case_name}", case.runTest)
