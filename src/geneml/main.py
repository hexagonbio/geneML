import gc
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import enlighten
from Bio.Seq import reverse_complement
from helperlibs.bio import seqio

from geneml.args import parse_args
from geneml.logger import setup_logger, write_setup_info
from geneml.outputs import build_cds_sequences, build_prediction_scores_seg, write_fasta, write_gff_file
from geneml.parallelism import compute_optimal_num_parallelism
from geneml.params import Params, Strand, build_params_namedtuple
from geneml.produce_genes import (
    Transcript,
    assign_transcripts_to_genes,
    build_transcripts,
    filter_by_dynamic_threshold,
    run_model,
)

logger = logging.getLogger("geneml")


def parse_contigs(inpath: str, contigs_filter: list[str] | None) -> tuple[dict[str, str], int]:
    """Parse and validate contig sequences from an input file.

    Reads genome records, restricted to IDs in contigs_filter if provided.
    Sequences are converted to uppercase and validated to contain only valid nucleotide characters.

    Args:
        inpath: Path to an input sequence file in FASTA/GenBank/EMBL format
        contigs_filter: Optional list of contig IDs to include

    Returns:
        Tuple of (contigs, genome_size) where contigs is a dictionary mapping contig ID to
        sequence and genome_size is the total number of bases
    """
    contigs = {}
    genome_size = 0
    to_process = set(contigs_filter or [])

    for record in seqio.parse(inpath):
        if contigs_filter and record.id not in to_process:
            # Skip this record
            continue
        to_process.discard(record.id)

        seq = str(record.seq).upper()
        # Check if sequence is valid
        if not seq:
            raise ValueError(f"Contig {record.id} has no sequence.")
        invalid_chars = set(seq) - set('ACGTN')
        if invalid_chars:
            raise ValueError(
                f"Sequence of contig {record.id} contains invalid characters: "
                f"{', '.join(sorted(invalid_chars))}."
            )

        contigs[record.id] = seq
        genome_size += len(seq)

    if to_process:
        raise ValueError(
            f"The following contig IDs were specified with --contigs-filter "
            f"but not found in input file: {', '.join(sorted(to_process))}"
        )

    return contigs, genome_size


def process_contig(contig_id: str, seq: str, params: Params, tensorflow_thread_count=None) -> tuple[str, list[Transcript], str | None]:
    """Process one contig and return transcript predictions or raw score segments.

    Returns only Python-native structures so the result is safely picklable
    across process boundaries.

    Args:
        contig_id: Contig identifier.
        seq: Contig DNA sequence.
        params: Runtime parameters controlling prediction behavior.
        tensorflow_thread_count: Optional thread count for TensorFlow ops.

    Returns:
        Tuple of contig ID, transcript list, and optional SEG-formatted scores.
    """
    start_time = time.time()

    import tensorflow as tf

    from geneml.model_loader import get_cached_gene_ml_model

    if params.cpu_only:
        tf.config.set_visible_devices([], 'GPU')

    tf.config.threading.set_inter_op_parallelism_threads(tensorflow_thread_count)
    tf.config.threading.set_intra_op_parallelism_threads(tensorflow_thread_count)

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


def process_genome(params: Params) -> None:
    """Run the end-to-end prediction pipeline for one input genome.

    Args:
        params: Runtime parameters controlling model execution and outputs.

    Returns:
        None.
    """
    num_cores = params.num_cores
    genome_start_time = time.time()

    contigs, genome_size = parse_contigs(params.inpath, params.contigs_filter)

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

    if num_cores is None:
        num_cores, tensorflow_thread_count = compute_optimal_num_parallelism(num_contigs=len(contigs))
        logger.info('Based on available memory, setting parallelism to %d parallel processes '
                    'and tensorflow threads to %s',
                    num_cores, tensorflow_thread_count or 'all available')
    else:
        tensorflow_thread_count = None

    reordered_contigs = reorder_contigs(contigs, num_cores)

    transcripts_by_contig_id = {}
    all_segs = []
    manager = enlighten.get_manager()
    progress = manager.counter(desc=f'Processing {params.inpath}', total=genome_size, unit='bp', color='green')
    if num_cores == 1:
        logger.info('Running from main thread, parallelism only for tensorflow')
        for contig_id, seq in reordered_contigs:
            _, r, segs = process_contig(contig_id, seq, params, tensorflow_thread_count)
            seq_len = len(contigs[contig_id])
            progress.update(seq_len)
            transcripts_by_contig_id[contig_id] = r
            if segs:
                all_segs.append(segs)
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            future_to_contig = {}
            for contig_id, seq in reordered_contigs:
                future = pool.submit(process_contig, contig_id, seq, params, tensorflow_thread_count)
                future_to_contig[future] = contig_id

            for future in as_completed(future_to_contig):
                contig_id = future_to_contig[future]
                seq_len = len(contigs[contig_id])
                progress.update(seq_len)
                _, r, segs = future.result()
                transcripts_by_contig_id[contig_id] = r
                if segs:
                    all_segs.append(segs)

    logger.info('Finished processing all contigs')
    if params.dynamic_scoring:
        logger.info('Filtering gene calls by dynamic threshold')
        transcripts_by_contig_id = filter_by_dynamic_threshold(transcripts_by_contig_id,
                                                               params.min_gene_score,
                                                               params.basepath)
    genes_by_contig_id, mean_gene_score = assign_transcripts_to_genes(transcripts_by_contig_id,
                                                                      params.gene_id_prefix)

    if all_segs:
        logger.info('Writing raw scores to %s', params.basepath+'.seg')
        with open(params.basepath+'.seg', 'w', encoding='utf-8') as f:
            f.write('#track graphType=heatmap maxHeightPixels=20:20:20 color=0,0,255 altColor=255,0,0\n')
            for segs in all_segs:
                f.write(f'{segs}\n')
    else:
        logger.info('Writing gene predictions to %s', params.outpath)
        write_gff_file(contigs, genes_by_contig_id, params.outpath,
                   mean_gene_score=mean_gene_score)
        if params.output_genes or params.output_proteins:
            cdses_by_transcript = build_cds_sequences(contigs, genes_by_contig_id)
            if params.output_genes:
                logger.info('Writing gene sequences to %s', params.output_genes)
                write_fasta(cdses_by_transcript, params.output_genes, sequence_type = 'cds')
            if params.output_proteins:
                logger.info('Writing protein sequences to %s', params.output_proteins)
                write_fasta(cdses_by_transcript, params.output_proteins, sequence_type = 'protein')

    elapsed = time.time() - genome_start_time
    logger.info('Finished processing %s, %.1fMB, in %.2f minutes',
                params.inpath, genome_size/1e6, elapsed/60)


def main() -> None:
    """Parse CLI arguments, configure logging, and run genome processing.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    params = build_params_namedtuple(args)

    if params.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    logfile = ''.join([params.basepath, '.log'])
    setup_logger(logfile, debug = params.debug, verbose= params.verbose)
    write_setup_info(params)
    process_genome(params)


if __name__ == "__main__":
    main()
