from __future__ import annotations


class AsterinisError(Exception):
    """
    Base exception for all errors raised by Asterinis.

    Applications can catch this exception when they want to handle
    framework-level failures without depending on a specific subsystem.
    """


class ConnectorError(AsterinisError):
    """Raised when a connector fails to execute."""


class PipelineError(AsterinisError):
    """Raised when a pipeline step fails."""


class ProviderError(AsterinisError):
    """Raised when a provider fails to process a request."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider exceeds its configured timeout."""


class RoutingError(AsterinisError):
    """Raised when a request cannot be routed correctly."""


class RetrievalError(AsterinisError):
    """Raised when retrieval fails or produces an invalid result."""


class AgentError(AsterinisError):
    """Raised when an agent fails during execution."""


class SecurityError(AsterinisError):
    """Raised when an operation violates a security policy."""


class ConfigurationError(AsterinisError):
    """Raised when Asterinis configuration is invalid."""
