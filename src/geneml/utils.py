import multiprocessing
import numpy as np
import psutil

from geneml.model_loader import ResidualModelBase


def chunked_seq_predict(model: ResidualModelBase, seq: str, chunk_size=100000, padding=1000):
    """
    Predicts the sequence in chunks of a given size to control memory usage, with some padding to handle
    the sequence context
    """
    seq = seq.upper()
    seq_len = len(seq)
    pred_list = []
    for i in range(0, seq_len, chunk_size):
        start = i
        end = min(seq_len, i + chunk_size)
        padded_start = max(0, start - padding)
        padded_end = min(seq_len, end + padding)
        preds = model.predict(seq[padded_start:padded_end], return_dict=False)
        pred_list.append(preds[:, (start-padded_start):(end-padded_start)])
    return np.concatenate(pred_list, axis=1)


def compute_optimal_num_parallelism(num_contigs) -> tuple[int, int | None]:
    """As the amount of memory increases, we can be more aggressive with the number of parallel processes since
    there is a larger shared buffer to handle memory spikes within a given subprocess/contig."""
    gb_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)

    if gb_available < 8:
        print(f'WARNING: Detected a low available memory environment: {gb_available} GB available. Setting num_cores to 1.')
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
