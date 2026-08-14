"""Behavioral compatibility primitives cross-validated against MATLAB."""

from .canny import matlab_canny_compat
from .clahe import matlab_adapthisteq_compat
from .morphology import (
    matlab_branchpoints_compat,
    matlab_closing_compat,
    matlab_spur_compat,
    matlab_thicken_compat,
    matlab_thin_compat,
)

__all__ = [
    "matlab_adapthisteq_compat",
    "matlab_branchpoints_compat",
    "matlab_canny_compat",
    "matlab_closing_compat",
    "matlab_spur_compat",
    "matlab_thicken_compat",
    "matlab_thin_compat",
]
