"""Common definitions for the simulate module.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum


class Sender(enum.Enum):
    """Type of sender."""

    PAYLOAD_LIMITED_SENDER = enum.auto()
    """Payload-limited sender."""
    DISTORTION_LIMITED_SENDER = enum.auto()
    """Distortion-limited sender."""
    PAYLOAD_LIMITED_SENDER_DDE = enum.auto()
    """Payload-limited sender -- fallback to DDE."""
