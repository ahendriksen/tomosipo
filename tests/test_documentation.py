#!/usr/bin/env python3

import pytest
from sphinx.application import Sphinx


@pytest.mark.slow
def test_html_docs():
    source_dir = u'doc'
    config_dir = u'doc'
    output_dir = u'doc/_build'
    doctree_dir = u'doc/_build/doctrees'
    all_files = 1

    app = Sphinx(
        source_dir,
        config_dir,
        output_dir,
        doctree_dir,
        buildername='html',
        warningiserror=False,
    )
    app.build(force_all=all_files)
