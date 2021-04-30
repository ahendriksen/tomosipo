import unittest
import pytest
import tomosipo as ts
from . import add_doctest_cases


# The tests are included in the doc strings of the methods.
class TestDocs(unittest.TestCase):
    pass


add_doctest_cases(TestDocs, ts.types)
