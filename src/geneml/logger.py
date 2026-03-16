import logging
import sys

from geneml import __version__

logger = logging.getLogger("geneml")


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback) -> None:
    """Log uncaught exceptions while preserving keyboard interrupt behavior.

    Args:
        exc_type: Exception type.
        exc_value: Exception instance.
        exc_traceback: Exception traceback object.

    Returns:
        None.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_uncaught_exceptions


def setup_logger(logfile, debug = False, verbose = False) -> None:
    """Configure file and stream handlers for the project logger.

    Args:
        logfile: Output path for the log file.
        debug: Whether to set console logging to DEBUG level.
        verbose: Whether to set console logging to INFO level.

    Returns:
        None.
    """
    log_format = '%(levelname)-8s %(asctime)s   %(message)s'
    date_format = "%d/%m %H:%M:%S"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_level_logfile = logging.INFO
    log_level_stdout = logging.WARNING
    if debug:
        log_level_stdout = logging.DEBUG
    elif verbose:
        log_level_stdout = logging.INFO

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(log_level_logfile)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level_stdout)

    formatter = logging.Formatter(log_format, datefmt=date_format)
    for handler in file_handler, stream_handler:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def write_setup_info(params) -> None:
    """Write startup metadata and parameter configuration to the log.

    Args:
        params: Runtime parameters object.

    Returns:
        None.
    """
    logger.info("Running geneML version %s", __version__)
    logger.info("Command line: %s", " ".join(sys.argv[1:]))
    parameter_info = '\n'.join(["Parameters:", params.to_log_json(indent=2)])
    logger.info(parameter_info)
