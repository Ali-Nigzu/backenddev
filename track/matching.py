"""Compatibility exports for Track V2 candidate sorting helpers.

The production reducer no longer runs matching through this module. Candidate
building and assignment are split between ``candidate_builder`` and
``assignment`` so internals consume only the normalized private configuration.
"""

from track.assignment import candidate_sort_key
from track.candidate_builder import observation_sort_key, track_sort_key

__all__ = ["candidate_sort_key", "observation_sort_key", "track_sort_key"]
