class IvoryError(Exception):
    """Base class for Ivory specific errors."""


class EarlyStopped(IvoryError):
    """Exception for early stopped runs.

    This error tells a trainer that the current `run` was early stopped.
    """
