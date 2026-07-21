class Parcel2dModflowError(RuntimeError):
    """Base class for exceptions in this module."""


class ValidationError(Parcel2dModflowError):
    """Exception raised for errors in the input validation."""


class ConfigError(ValidationError):
    """Exception raised for errors in the configuration."""


class MissingDataError(Parcel2dModflowError):
    """Exception raised for missing data errors."""
