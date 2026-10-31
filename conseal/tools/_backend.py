"""Backend-related definitions and routines.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum


class Backend(enum.Enum):
    """Backend implementation to choose from."""

    BACKEND_PYTHON = enum.auto()
    """Python (baseline) implementation."""
    BACKEND_RUST = enum.auto()
    """Rust implementation, default.

    Falls back to Python with a warning, where not supported."""

    def __enter__(self):
        global backend
        self._backend_prev = backend
        backend = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global backend
        backend = self._backend_prev


backend = Backend.BACKEND_RUST


def get_backend() -> Backend:
    """Returns the current backend.

    :return: current backend, Rust by default
    :rtype: :class:`Backend`

    :Example:

    >>> cl.get_backend()
    <Backend.BACKEND_RUST: 2>
    """
    return backend


def set_backend(b: Backend):
    """Sets the backend globally.

    To set it only for a block of code, use the backend as a context manager.

    :param b: backend to use
    :type b: :class:`Backend`

    :Example:

    >>> cl.set_backend(cl.BACKEND_PYTHON)
    >>> with cl.BACKEND_RUST:
    ...     rhos = cl.hill.compute_cost_adjusted(x0)
    """
    global backend
    backend = b
