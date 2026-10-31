"""Sphinx role linking a method to the unit test against its reference implementation.

Usage, with a reference key and a pytest-like test id::

    :verified:`binghamton test/test_wow.py::TestWOW::test_simulate`

renders a badge, linking to the lines of the test on GitHub,
followed by the key of the reference implementation.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import ast
import html
import os
from pathlib import Path
import subprocess

from docutils import nodes
from sphinx.util import logging

logger = logging.getLogger(__name__)

REPOSITORY = 'https://github.com/uibk-uncover/conseal'
ROOT = Path(__file__).resolve().parents[2]

# reference implementations, key: (key shown, tooltip)
REFERENCES = {
    'binghamton': ('Binghamton', 'Matlab by DDE Lab, Binghamton University'),
    'troyes': ('Troyes', 'Matlab by R. Cogranne, Troyes University of Technology'),
    'barcelona': ('Barcelona', 'Python by D. Lerch Hostalot, Universitat Oberta de Catalunya'),
    'lille': ('Lille', 'Python by P. Bas, CNRS, University of Lille'),
}


def _git(*args) -> str:
    # not stripped, leading blank lines of a file count
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def _commit() -> str:
    """Commit to link to, None for the working tree of Read the Docs."""
    # Read the Docs builds a pushed commit
    if os.environ.get('READTHEDOCS_GIT_COMMIT_HASH'):
        return None
    # local build links to the last pushed commit,
    # as local commits do not exist on GitHub
    try:
        return _git('merge-base', 'HEAD', '@{upstream}').strip()
    except (OSError, subprocess.CalledProcessError):
        return 'main'


def _source(path: str, commit: str) -> str:
    """Source of the file, at the commit linked to."""
    if commit is None:
        return (ROOT / path).read_text()
    return _git('show', f'{commit}:{path}')


def _lines(source: str, path: str, names: list) -> tuple:
    """First and last line of the function, decorators included."""
    tree = ast.parse(source)
    body = tree.body
    for name in names:
        node = next(
            (n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name),
            None,
        )
        if node is None:
            raise LookupError(f'{name} not found in {path}')
        body = node.body
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, node.end_lineno


def verified_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Badge linking to the test against the reference implementation."""
    # parse key and test id, the test must exist in the working tree
    try:
        key, test = text.split()
        path, *names = test.split('::')
        shown, reference = REFERENCES[key]
        _lines((ROOT / path).read_text(), path, names)
    except (ValueError, KeyError, LookupError, OSError) as e:
        msg = inliner.reporter.warning(f'verified: invalid {text!r}: {e}', line=lineno)
        return [inliner.problematic(rawtext, rawtext, msg)], [msg]

    # lines of the test at the commit linked to,
    # whole file, if the test is not pushed yet
    commit = _commit()
    url = f'{REPOSITORY}/blob/{commit or os.environ["READTHEDOCS_GIT_COMMIT_HASH"]}/{path}'
    try:
        start, end = _lines(_source(path, commit), path, names)
        url += f'#L{start}-L{end}'
    except (LookupError, subprocess.CalledProcessError, SyntaxError):
        logger.info(f'verified: {test} not at commit {commit}, linking the file')

    # badge to the test, key to the footnote
    title = html.escape(f'Stego identical to {reference}, see {names[-1]} in {path}')
    badge = (
        f'<a class="verified" href="{url}" title="{title}" '
        f'target="_blank" rel="noopener noreferrer">unit test</a>'
        f' <a class="verified-ref" href="#ref-{key}">{shown}</a>'
    )
    return [nodes.raw('', badge, format='html')], []


def setup(app):
    app.add_role('verified', verified_role)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
