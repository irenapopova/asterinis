class AsterinisError(Exception):
    """Base exception for Asterinis."""


class RoutingError(AsterinisError):
    """Raised when routing cannot be completed."""


class ConnectorError(AsterinisError):
    """Raised when a connector fails."""


class PipelineError(AsterinisError):
    """Raised when a pipeline step fails."""