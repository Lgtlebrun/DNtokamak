"""Fits of the current configuration to a target separatrix shape."""

from dntokamak.optimize import symmetric, up_down
from dntokamak.optimize.common import Result

__all__ = ["Result", "symmetric", "up_down"]
