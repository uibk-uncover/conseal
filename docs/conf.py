# Configuration file for the Sphinx documentation builder.

# -- Project information
project = 'conseal'
copyright = '2024–2026, Martin Benes & Benedikt Lorch, University of Innsbruck'
author = 'Martin Benes, Benedikt Lorch'

release = '2026.11'
version = '2026.11'

# -- General configuration

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / '_ext'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_design',
    'verified',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
intersphinx_disabled_domains = ['std']

# section labels unique across pages, e.g., guide/dataset:Parallel processing
autosectionlabel_prefix_document = True

templates_path = ['templates']

# -- Options for HTML output
html_theme = 'pydata_sphinx_theme'
html_static_path = ['static']
html_css_files = ['css/custom.css']
html_js_files = ['js/custom.js']
html_title = 'conseal'
html_logo = 'static/seal.png'
html_favicon = 'static/seal.png'
html_theme_options = {
    'logo': {
        'text': 'conseal',
    },
    'github_url': 'https://github.com/uibk-uncover/conseal',
    'icon_links': [
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/conseal/',
            'icon': 'fa-brands fa-python',
        },
    ],
    'navbar_align': 'left',
    'show_toc_level': 2,
    'navigation_with_keys': False,
    'footer_start': ['copyright'],
    'footer_end': [],
}
html_show_sourcelink = False

# no section navigation on pages without subpages
html_sidebars = {
    'faq': [],
    'glossary': [],
}

# copy code without the >>> and ... prompts
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True

# -- Options for EPUB output
epub_show_urls = 'footnote'


# -- Module docstrings

def _strip_authors(app, what, name, obj, options, lines):
    """Drops Author and Affiliation lines from module docstrings."""
    if what == 'module':
        lines[:] = [l for l in lines if not l.startswith(('Author:', 'Affiliation:'))]


def setup(app):
    app.connect('autodoc-process-docstring', _strip_authors)
