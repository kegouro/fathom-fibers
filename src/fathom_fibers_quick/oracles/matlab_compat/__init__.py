"""Behavioral compatibility primitives cross-validated against MATLAB."""

from .clahe import matlab_adapthisteq_compat
from .canny import matlab_canny_compat

__all__ = ["matlab_adapthisteq_compat", "matlab_canny_compat"]
