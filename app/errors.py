class TranscriptionError(Exception):
    """Base class for all transcription pipeline errors."""


class ProviderError(TranscriptionError):
    """A single provider failed to produce a transcript."""


class AllProvidersFailedError(TranscriptionError):
    """Every configured provider failed."""


class AudioValidationError(TranscriptionError):
    """The uploaded audio failed validation (missing, too large, unsupported)."""


class UnsupportedFormatError(AudioValidationError):
    """The audio format/extension is not accepted."""
