"""Parallel processing utilities for geneML."""

import logging
import os

import psutil

from geneml.logger import setup_worker_logger
from geneml.tf_setup import configure_threads, import_tensorflow

logger = logging.getLogger("geneml")


def available_cpus() -> int:
    """Find the number of available CPUs on this system.

    Args:
        None.

    Returns:
        The number of available CPUs.
    """
    n = None
    if hasattr(os, "process_cpu_count"):
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    if not n:
        logger.warning("Could not determine the number of available CPUs; defaulting to 1.")
        return 1
    return n


def resolve_cpu_budget(user_cores: int | None) -> int:
    """Resolve the number of CPU cores to use for parallel processing.

    Args:
        user_cores: The number of cores requested by the user.

    Returns:
        The number of CPU cores to use.
    """
    available = available_cpus()
    logger.info("Detected %d available CPUs.", available)

    if user_cores is None:
        return available
    if user_cores > available:
        logger.warning("%d cores requested, but only %d are available; using %d.",
                       user_cores, available, available)
    return max(1, min(user_cores, available))


def available_memory_gb() -> float:
    """Find the amount of available memory on this system in GB.

    Args:
        None.

    Returns:
        The amount of available memory in GB.
    """
    return psutil.virtual_memory().available / 1024**3



def memory_limited_processes(mem_per_process_gb) -> int:
    """Compute the number of processes that can be run in parallel based on available memory.

    Args:
        mem_per_process_gb: The amount of memory required per process in GB.

    Returns:
        The number of processes that can be run in parallel.
    """
    gb = available_memory_gb()
    n = int(gb / mem_per_process_gb)
    if n < 1:
        logger.warning("Only %.1f GB of memory available; a single process may need around %.1f GB.",
                       gb, mem_per_process_gb)
        return 1
    return n



def compute_num_processes(cpu_budget: int, num_contigs: int, user_cores: int | None) -> int:
    """Compute the number of processes to use for parallel processing.

    If the user has specified a number of cores, use that number.
    Otherwise, use the minimum of available CPUs and the number of processes that can be run
    based on available memory.
    Processes cannot exceed the number of contigs.

    Args:
        cpu_budget: The number of CPU cores available for parallel processing.
        num_contigs: The number of contigs to process.
        user_cores: The number of cores requested by the user.

    Returns:
        The number of processes to use for parallel processing.
    """
    by_memory = memory_limited_processes(mem_per_process_gb=2.5)
    if user_cores is not None:
        if user_cores > by_memory:
            logger.warning("%d processes requested, but available memory only supports %d.",
                           user_cores, by_memory)
        processes = max(1, min(user_cores, num_contigs))
    else:
        processes = max(1, min(by_memory, cpu_budget, num_contigs))
    return processes


def compute_tensorflow_threads(cpu_budget: int, num_processes: int) -> int:
    """Compute the number of threads to use for TensorFlow.

    Args:
        cpu_budget: The number of CPU cores available for parallel processing.
        num_processes: The number of processes to use for parallel processing.

    Returns:
        The number of threads to use for TensorFlow.
    """
    assert num_processes > 0, "Number of processes must be greater than 0."
    assert cpu_budget > 0, "CPU budget must be greater than 0."
    return max(1, cpu_budget // num_processes)


def init_worker(tf_threads: int, log_queue) -> None:
    """Initialize a worker process for parallel processing.

    Args:
        tf_threads: The number of threads to use for TensorFlow.
        log_queue: A multiprocessing.Queue for logging.

    Returns:
        None.
    """
    setup_worker_logger(log_queue)
    import_tensorflow()
    configure_threads(tf_threads)
