"""superfloat.arduino: integer-only SuperFloat inference for edge devices."""

from .formats import SF4, SF8, SF16, SFFormat, get_format

__all__ = ["SF4", "SF8", "SF16", "SFFormat", "get_format"]
__version__ = "0.1.0"
