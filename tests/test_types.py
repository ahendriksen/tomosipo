import unittest
import pytest
import tomosipo as ts
from . import add_doctest_cases, UniformPrintingTestCase


# Test the doctests that are included in the docstrings of ts.types
class TestDocs(UniformPrintingTestCase):
    pass


add_doctest_cases(TestDocs, ts.types)
