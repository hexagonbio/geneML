"""Main entry point for geneML."""

import logging
import sys

from geneml.args import parse_args
from geneml.logger import setup_logger, write_setup_info
from geneml.params import build_params_namedtuple
from geneml.tf_setup import import_tensorflow

logger = logging.getLogger("geneml")


def main() -> None:
    """Parse CLI arguments, configure logging, and run genome processing.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    params = build_params_namedtuple(args)

    logfile = ''.join([params.basepath, '.log'])
    setup_logger(logfile, debug = params.debug, verbose= params.verbose)
    write_setup_info(params)

    try:
        import_tensorflow(params.cpu_only)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    from geneml.run import run_geneML
    run_geneML(params)


if __name__ == "__main__":
    main()
