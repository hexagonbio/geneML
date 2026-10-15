"""End-to-end pipeline for running gene prediction on a genome."""

import gc
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import tensorflow as tf
from Bio.Seq import reverse_complement

from geneml.model_loader import get_cached_gene_ml_model
from geneml.outputs import build_prediction_scores_seg
from geneml.parallelism import compute_num_processes, compute_tensorflow_threads, resolve_cpu_budget
from geneml.params import Params, Strand
from geneml.produce_genes import (
    assign_transcripts_to_genes,
    build_transcripts,
    filter_by_dynamic_threshold,
    run_model,
)
from geneml.types import GenomeResult, Transcript

logger = logging.getLogger("geneml")


def process_contig(contig_id: str, seq: str, params: Params) -> tuple[str, list[Transcript], str | None]:
    """Process one contig and return transcript predictions or raw score segments.

    Returns only Python-native structures so the result is safely picklable
    across process boundaries.

    Args:
        contig_id: Contig identifier.
        seq: Contig DNA sequence.
        params: Runtime parameters controlling prediction behavior.

    Returns:
        Tuple of contig ID, transcript list, and optional SEG-formatted scores.
    """
    start_time = time.time()

    model = get_cached_gene_ml_model(params.model_path, params.context_length)

    logger.info('Processing contig %s of size %d', contig_id, len(seq))

    preds = None
    rc_preds = None
    rc_seq = None
    if params.strand is not Strand.REVERSE:
        logger.info('%s 1/5: Running model on forward strand', contig_id)
        preds = run_model(model, seq, params.context_length)

    if params.strand is not Strand.FORWARD:
        logger.info('%s 2/5: Running model on reverse strand', contig_id)
        rc_seq = reverse_complement(seq)
        rc_preds = run_model(model, rc_seq, params.context_length)

    if params.output_segs:
        segs = str(build_prediction_scores_seg(contig_id, preds, rc_preds))
        return contig_id, [], segs

    transcripts = build_transcripts(preds, rc_preds, seq, rc_seq, contig_id, params)

    # explicitly clean up memory after finishing a contig
    del preds
    del rc_preds
    gc.collect()

    elapsed = time.time() - start_time
    logger.info('Finished processing contig %s in %.2f seconds, %.2f bp/s',
                contig_id, elapsed, len(seq)/elapsed)

    return contig_id, transcripts, None


def reorder_contigs(contigs, num_cores) -> list[tuple[str, str]]:
    """Reorder contigs to better balance parallel work.

    Args:
        contigs: Mapping of contig IDs to sequences.
        num_cores: Number of worker processes.

    Returns:
        Ordered list of (contig_id, sequence) tuples.
    """
    contigs_by_size = sorted(contigs.items(), key=lambda x: len(x[1]), reverse=False)
    if len(contigs_by_size) < num_cores * 2:
        return contigs_by_size

    reordered_contigs = []
    num_groups = max(num_cores, 8)
    offset = len(contigs_by_size) // num_groups + 1
    for i in range(offset):
        for j in range(0, len(contigs_by_size), offset):
            if j + i < len(contigs_by_size):
                reordered_contigs.append(contigs_by_size[j + i])
    assert len(reordered_contigs) == len(contigs_by_size), f'failed to reorder contigs, {len(reordered_contigs)} != {len(contigs_by_size)}'
    return reordered_contigs


def process_genome(contigs: dict[str, str], params: Params,
                   on_contig_done = None) -> GenomeResult:
    """Run the end-to-end prediction pipeline for one input genome.

    Args:
        contigs: Mapping of contig IDs to sequences.
        params: Runtime parameters controlling model execution and outputs.
        on_contig_done: Optional callback function to be called when each contig is done processing.

    Returns:
        GenomeResult containing the predicted genes and segment information.
    """
    user_cores = params.num_cores

    num_contigs = len(contigs)
    contig_order = list(contigs.keys())
    genome_size = sum(len(seq) for seq in contigs.values())

    # Disable dynamic scoring if the input sequence is too short
    if params.dynamic_scoring and genome_size < 100_000:
        logger.warning(
            'Input sequence is too small (%d bp) for dynamic scoring. '
            'Using fixed threshold of %.2f instead. '
            'Consider specifying a custom threshold with --min-gene-score.',
            genome_size, params.min_gene_score
        )
        # Create new Params with dynamic_scoring disabled
        params = params._replace(dynamic_scoring=False)

    cpu_budget = resolve_cpu_budget(user_cores)
    num_processes = compute_num_processes(cpu_budget, num_contigs, user_cores)
    tf_threads = compute_tensorflow_threads(cpu_budget, num_processes)

    logger.info("Running %d parallel processes.", num_processes)
    logger.info("Running TensorFlow with %d threads per process.", tf_threads)

    tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    reordered_contigs = reorder_contigs(contigs, num_processes)

    transcripts_by_contig_id = {}
    segs_by_contig_id = {}

    if num_processes == 1:
        for contig_id, seq in reordered_contigs:
            _, r, segs = process_contig(contig_id, seq, params)
            if on_contig_done:
                on_contig_done(contig_id, len(seq))
            transcripts_by_contig_id[contig_id] = r
            if segs:
                segs_by_contig_id[contig_id] = segs
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            future_to_contig = {}
            for contig_id, seq in reordered_contigs:
                future = pool.submit(process_contig, contig_id, seq, params)
                future_to_contig[future] = contig_id

            for future in as_completed(future_to_contig):
                contig_id = future_to_contig[future]
                if on_contig_done:
                    on_contig_done(contig_id, len(contigs[contig_id]))
                _, r, segs = future.result()
                transcripts_by_contig_id[contig_id] = r
                if segs:
                    segs_by_contig_id[contig_id] = segs

    # Reorder transcripts and segs to match original contig order
    transcripts_by_contig_id = {
        contig_id: transcripts_by_contig_id[contig_id]
        for contig_id in contig_order
        if contig_id in transcripts_by_contig_id
    }
    all_segs = [segs_by_contig_id[contig_id] for contig_id in contig_order
                if contig_id in segs_by_contig_id]

    logger.info('Finished processing all contigs')
    if params.dynamic_scoring:
        logger.info('Filtering gene calls by dynamic threshold')
        transcripts_by_contig_id = filter_by_dynamic_threshold(transcripts_by_contig_id,
                                                               params.min_gene_score)
    genes_by_contig_id = assign_transcripts_to_genes(transcripts_by_contig_id,
                                                     params.gene_id_prefix)

    return GenomeResult(genes_by_contig_id=genes_by_contig_id,
                        segs=all_segs)
