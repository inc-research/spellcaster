"""Spellcaster exception hierarchy."""


class SpellcasterError(Exception):
    """Base exception for all spellcaster errors."""


class InsufficientDataError(SpellcasterError):
    """Raised when input data is too small or empty for meaningful analysis."""


class ModelNotLoadedError(SpellcasterError):
    """Raised when a required NLP model is not available."""


class OptionalDependencyError(SpellcasterError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, package: str, extra: str):
        self.package = package
        self.extra = extra
        super().__init__(
            f"'{package}' is required for this feature. "
            f"Install it with: pip install spellcaster[{extra}]"
        )
