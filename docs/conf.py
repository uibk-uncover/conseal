# Configuration file for the Sphinx documentation builder.

# -- Project information

import sphinx_rtd_theme

project = 'conseal'
copyright = '2023, Martin Benes & Benedikt Lorch, University of Innsbruck'
author = 'Martin Benes, Benedikt Lorch'

release = '2023.12'
version = '2023.12.a0'

# -- General configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'alabaster'  # 'sphinx_rtd_theme'
html_theme_options = {
    'logo': 'seal.png',
    'logo_name': 'conseal',
    'description': 'A Python package, implementing modern image steganographic algorithms.',
    'fixed_sidebar': True,
    'github_button': True,
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
