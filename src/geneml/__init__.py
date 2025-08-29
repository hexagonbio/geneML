import os

__version__ = "0.1.0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow logging (1: errors, 2: warnings, 3: info)
