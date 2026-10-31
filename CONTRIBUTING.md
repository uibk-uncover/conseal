# Contributing to conseal

Thank you for considering a contribution!
conseal implements steganographic methods for research,
so we care most about correctness and reproducibility.

You can contribute by

- reporting a bug or a deviation from a reference implementation, as a [GitHub issue](https://github.com/uibk-uncover/conseal/issues),
- implementing a new method, or fixing an existing one,
- improving the documentation.

We accept only methods supported by peer-reviewed research.


## Development setup

conseal is a Python package with a Rust extension,
so you need the [Rust toolchain](https://rustup.rs/) in addition to Python.

```bash
git clone https://github.com/uibk-uncover/conseal
cd conseal
python -m venv .venv
source .venv/bin/activate
pip install maturin
pip install -e '.[dev]'
```

After changing the Rust code in `src/`, rebuild the extension.

```bash
maturin develop --release
```


## Tests

Run the tests from the root of the repository.

```bash
python -m unittest                    # all tests, as in the CI
python -m unittest test.test_hill     # tests of a single method
python -m unittest -k _reference      # tests against reference implementations
```

The tests against reference implementations follow a common naming.

| Test | Compares |
| --- | --- |
| `test_stego_reference` | stego images, for the same seed |
| `test_cost_reference` | costs |
| `test_probability_reference` | change probabilities |

The documentation links these tests from the methods table,
so keep their names when you change them.


## Code quality

The CI checks the code with flake8.

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
```

Please

- keep the functional interface of the package, i.e., functions, not classes,
- document each public function with a docstring, with `:param:`, `:type:`, `:return:` and an example,
- comment the steps of the code, with a short comment above the lines.


## Adding a method

A method lives in its own module, e.g., `conseal/hill/`, with

- `_costmap.py`, with `compute_cost` (raw costs) and `compute_cost_adjusted` (costs with wet elements),
- `_simulate.py`, with `simulate_single_channel`, calling `conseal.simulator`,
- `__init__.py`, exporting the public functions.

Test the method against a reference implementation,
ideally the one of its authors, at the level of stego images.
If the reference has bugs, keep the original behavior as an `Implementation` option,
and add a fixed one, see the [reproducibility guide](https://conseal.readthedocs.io/en/latest/guide/reproducibility.html).

Do not commit the code of a reference implementation, unless its license allows it.
Commit its outputs, e.g., stego images, to `test/assets/`,
and describe in the test how they were produced.

Finally, add the method to the methods table on the title page, `docs/index.rst`,
with its paper and the `:verified:` role linking its reference test,
and to `docs/reference.rst`.


## Documentation

The documentation is built with Sphinx.

```bash
pip install -r docs/requirements.txt sphinx-autobuild
sphinx-autobuild docs docs/_build/html   # live preview at http://127.0.0.1:8000
```

The build must pass without warnings, as on Read the Docs.

```bash
sphinx-build -W -b html docs docs/_build/html
```


## Pull requests

Open the pull request against the `main` branch.
Describe what you changed and why,
and how you checked it, e.g., against which reference.


## License

By contributing, you agree that your contribution is licensed
under the [Mozilla Public License 2.0](LICENSE), as the rest of the package.
