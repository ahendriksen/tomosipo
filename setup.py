#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = [
    'astra-toolbox>=2.0',
    'numpy',
]

setup_requirements = [
    'pytest-runner'
]

test_requirements = [
    'pytest'
]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'sphinx-autodoc-typehints',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
    'pytest',
    'pytest-runner'
]

setup(
    author="Allard Hendriksen",
    author_email='allard.hendriksen@cwi.nl',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPL-3.0-only',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="A usable Python astra-based tomography library.",
    install_requires=requirements,
    license="GPL-3.0-only",
    long_description=readme,
    include_package_data=True,
    keywords='tomography',
    name='tomosipo',
    packages=find_packages(),
    package_dir={'tomosipo': 'tomosipo'},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={'dev': dev_requirements},
    url='https://github.com/ahendriksen/tomosipo',
    # Also edit the version in tomosipo/__init__.py!
    version='0.4.1',
    zip_safe=False,
)
