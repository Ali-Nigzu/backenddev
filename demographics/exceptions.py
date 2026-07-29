"""Demographic stage exception types."""


class DemographicError(Exception):
    """Base error for the Demographic stage."""


class DemographicInputError(DemographicError):
    """Raised when EventBatch, FrameBatch, frames, or crops are malformed."""


class DemographicModelError(DemographicError):
    """Raised when model loading, checkpoint validation, or outputs are invalid."""
