"""Main module for tests.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import unittest

# logging
if __name__ == "__main__":
    import logging
    logging.basicConfig(filename="test.log", level=logging.INFO)
    import conseal
    logging.info(f"{conseal.__path__=}")
    logging.info(f"{conseal.__version__=}")

# === unit tests ===
from test_juniward import TestJUNIWARD  # noqa: F401,E402
from test_nsF5 import TestnsF5  # noqa: F401,E402
from test_simulate import TestSimulate  # noqa: F401,E402
from test_uerd import TestUERD  # noqa: F401,E402
# ==================

# run unittests
if __name__ == "__main__":
    unittest.main(verbosity=1)
