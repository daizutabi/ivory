class IvoryError(Exception):
    """Base class for Ivory specific errors."""


class EarlyStopped(IvoryError):
    """Exception for early stopped runs.

    This error tells a trainer that the current `run` should be early stopped.
    """


class Pruned(IvoryError):
    """Exception for pruned runs.

    This error tells a trainer that the current `run` should  be pruned.
    """


class TestDataNotFoundError(IvoryError):
    """Exception when test data not found."""
