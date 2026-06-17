import sys

from loguru import logger


def init_logger(sink=sys.stdout, **kwargs):
    """
    Initialize a logger using Loguru. By default, it will log to standard output, but
    you can specify a different sink (e.g., a file) and additional keyword arguments for
    configuration of loguru's logger.

    Parameters
    ----------
    sink : _type_, optional
        Where to log the messages. The default is sys.stdout.
    **kwargs
        Additional keyword arguments for configuring the logger (e.g., level, format).
        See Loguru's documentation for more details:
        https://loguru.readthedocs.io/en/stable/api/logger.html.

    """
    logger.remove()  # Remove the Loguru's default logger
    logger.add(sink, **kwargs)
