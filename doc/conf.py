# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from recommonmark.parser import CommonMarkParser
import doctest
import importlib

# -- Project information -----------------------------------------------------

project = 'tomosipo'
copyright = '2020, Allard Hendriksen'
author = 'Allard Hendriksen'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'recommonmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "./img/logo.svg"


# -- Support markdown -----------------------------------------------------
# See: <https://blog.readthedocs.com/adding-markdown-support/>

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Mock import modules if they are not otherwise available. This is needed for
# the test code to run properly.
def _is_package_available(package):
    try:
        importlib.import_module(package)
        return True
    except ModuleNotFoundError:
        return False

dependencies = ["cupy", "ffmpeg", "pyqtgraph", "torch", "odl"]
autodoc_mock_imports = [p for p in dependencies if not _is_package_available(p)]
autosummary_generate = True

napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

# Only include tests that are part of the documentation. Doctests in code
# comments are tested using pytest.
# To test the documentation source blocks, use:
# > python -msphinx -b doctest doc/ ./.doctest-output
doctest_test_doctest_blocks = ''

# Do not ignore exception details. See:
# https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html#confval-doctest_default_flags
doctest_default_flags = doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.ELLIPSIS

doctest_global_setup = f'''
import astra
cuda_available = astra.use_cuda()
matplotlib_available = {_is_package_available('matplotlib')}
torch_available = {_is_package_available('torch')}
'''
