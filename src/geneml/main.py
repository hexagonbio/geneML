"""Main entry point for geneML."""

import logging
import os

from geneml.args import parse_args
from geneml.logger import setup_logger, write_setup_info
from geneml.params import build_params_namedtuple

args = parse_args()
params = build_params_namedtuple(args)

if params.cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

os.environ["GLOG_minloglevel"] = "2"

# TF needs to be imported after environment variables are set
from geneml.run import run_geneML # noqa: E402, I001

logger = logging.getLogger("geneml")


def main() -> None:
    """Parse CLI arguments, configure logging, and run genome processing.

    Args:
        None.

    Returns:
        None.
    """
    logfile = ''.join([params.basepath, '.log'])
    setup_logger(logfile, debug = params.debug, verbose= params.verbose)
    write_setup_info(params)

    run_geneML(params)


if __name__ == "__main__":
    main()
