import os

from ._version import __version__

__all__ = ["__version__"]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow logging (1: errors, 2: warnings, 3: info)
