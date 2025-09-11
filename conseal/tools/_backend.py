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
    """RUST implementation, if available."""

    def __enter__(self):
        global backend
        self._backend_prev = backend
        backend = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global backend
        backend = self._backend_prev


backend = Backend.BACKEND_PYTHON


def get_backend() -> Backend:
    return backend


def set_backend(b: Backend):
    global backend
    backend = b
