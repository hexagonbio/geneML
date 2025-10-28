import logging
import multiprocessing

import psutil

logger = logging.getLogger("geneml")


def compute_optimal_num_parallelism(num_contigs) -> tuple[int, int | None]:
    """As the amount of memory increases, we can be more aggressive with the number of parallel processes since
    there is a larger shared buffer to handle memory spikes within a given subprocess/contig."""
    gb_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)

    if gb_available < 8:
        logger.warning('Detected a low available memory environment: %.2f GB available. '
                       'Setting num_cores to 1.',
                        gb_available)
        num_cores = 1
    elif gb_available < 32:
        gb_per_process = 5.0
        num_cores = int(gb_available / gb_per_process)
    else:
        gb_per_process = 3.5
        num_cores = int(gb_available / gb_per_process + 1)

    total_cores = multiprocessing.cpu_count()
    if num_cores >= total_cores:
        # if there are a good number of contigs and enough cores, set tensorflow to use a single thread so there's no contention
        tensorflow_thread_count = 1 if num_contigs > num_cores * 10 else None
        return total_cores, tensorflow_thread_count
    else:
        return num_cores, None
