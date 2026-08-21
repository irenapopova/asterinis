import logging


LOGGER_NAME = "asterinis"


def get_logger(name: str | None = None) -> logging.Logger:
    logger_name = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)