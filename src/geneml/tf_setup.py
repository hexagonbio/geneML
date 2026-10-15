"""Configure and import TensorFlow."""

import contextlib
import os


@contextlib.contextmanager
def suppress_stderr():
    """Silence the C-level stderr (fd 2), which sys.stderr redirection can't catch.

    This is used to suppress TensorFlow's verbose logging when running on CPU.

    Args:
        None.

    Returns:
        Context manager that silences stderr within its scope.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def import_tensorflow():
    """Import TensorFlow with CPU-only configuration and suppressed logging.

    Args:
        None.

    Returns:
        None.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    with suppress_stderr():
        import tensorflow as tf  # noqa: F401, PLC0415


def configure_threads(tf_threads: int) -> None:
    """Configure TensorFlow threading.

    Args:
        tf_threads: The number of threads to use for TensorFlow.

    Returns:
        None.
    """
    import tensorflow as tf  # noqa: F401, PLC0415
    tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
