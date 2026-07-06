"""Compatibility module for legacy imports.

The public V2 detection package exports only DetectV2 from detection.__init__.
New code should import DetectV2 from detection and execute it as a callable.
"""

from .detect_v2 import DetectV2

__all__ = ["DetectV2"]
