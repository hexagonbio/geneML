import argparse
import gc
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import enlighten
import tensorflow as tf
from Bio.Seq import reverse_complement
from helperlibs.bio import seqio

from geneml import __version__
from geneml.model_loader import get_cached_gene_ml_model
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

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_uncaught_exceptions

def setup_logger(logfile, debug = False, verbose = False):
    log_format = '%(levelname)-8s %(asctime)s   %(message)s'
    date_format = "%d/%m %H:%M:%S"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_level_logfile = logging.INFO
    log_level_stdout = logging.WARNING
    if debug:
        log_level_stdout = logging.DEBUG
    elif verbose:
        log_level_stdout = logging.INFO

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(log_level_logfile)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level_stdout)

    formatter = logging.Formatter(log_format, datefmt=date_format)
    for handler in file_handler, stream_handler:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def check_args(parser, args):
    if args.model and not args.context_length:
        parser.error("--context-length is required when using a custom model.")

def write_setup_info(params):
    logger.info("Running geneML version %s", __version__)
    logger.info("Command line: %s", " ".join(sys.argv[1:]))
    parameter_info = '\n'.join(["Parameters:", params.to_json(indent=2)])
    logger.info(parameter_info)


def process_contig(contig_id: str, seq: str, params: Params, tensorflow_thread_count=None) -> tuple[str, list[Transcript], str | None]:
    """
    Returns a python-only data structure so it can be pickled for either joblib or crossing over process boundaries
    """
    start_time = time.time()

    tf.config.threading.set_inter_op_parallelism_threads(tensorflow_thread_count)
    tf.config.threading.set_intra_op_parallelism_threads(tensorflow_thread_count)

    model = get_cached_gene_ml_model(params.model_path, params.context_length)

    logger.info('Processing contig %s of size %d', contig_id, len(seq))

    preds = None
    rc_preds = None
    rc_seq = None
    if params.strand is not Strand.REVERSE:
        logger.info('%s 1/5: Running model on forward strand', contig_id)
        preds = run_model(model, seq)

    if params.strand is not Strand.FORWARD:
        logger.info('%s 2/5: Running model on reverse strand', contig_id)
        rc_seq = reverse_complement(seq)
        rc_preds = run_model(model, rc_seq)

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
    """
    Reorders contigs by size to balance the workload across processes.
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


def process_genome(params: Params):
    num_cores = params.num_cores
    genome_start_time = time.time()

    contigs = {}
    genome_size = 0
    for record in seqio.parse(params.inpath):
        if params.contigs_filter is not None and record.id not in params.contigs_filter:
            continue
        seq = str(record.seq).upper()
        contigs[record.id] = seq
        genome_size += len(seq)

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
                                                               params.min_gene_score)
    genes_by_contig_id = assign_transcripts_to_genes(transcripts_by_contig_id)

    if all_segs:
        logger.info('Writing raw scores to %s', params.basepath+'.seg')
        with open(params.basepath+'.seg', 'w', encoding='utf-8') as f:
            f.write('#track graphType=heatmap maxHeightPixels=20:20:20 color=0,0,255 altColor=255,0,0\n')
            for segs in all_segs:
                f.write(f'{segs}\n')
    else:
        logger.info('Writing gene predictions to %s', params.outpath)
        write_gff_file(contigs, genes_by_contig_id, params.outpath)
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


def parse_args(argv=None):
    version_string = f"geneML {__version__}"
    parser = argparse.ArgumentParser(description=version_string, add_help=False)
    parser.add_argument('-h', '--help', action='help', help="Show this help message and exit.")
    parser.add_argument('--version', action='version', version=version_string, help="Show version number and exit.")
    parser.add_argument('sequence', type=str, help="Sequence file in FASTA/GenBank/EMBL format.")
    parser.add_argument('-o', '--output', type=str, help="Gene annotations output path (default: based on input filename).")
    parser.add_argument('-g', '--genes', type=str, help="Gene sequences output path (default: None).")
    parser.add_argument('-p', '--proteins', type=str, help="Protein sequences output path (default: None).")
    parser.add_argument('-m', '--model', type=str, help="Path to model file (default: models/geneML_default.keras).")
    parser.add_argument('-cl', '--context-length', type=int, help="Context length of the model.")
    parser.add_argument('-c', '--cores', type=int, help="Number of cores to use for processing (default: all available).")

    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument('-v', '--verbose', action='store_true', help="Enable verbose mode.")
    advanced.add_argument('-d', '--debug', action='store_true', help="Enable debug mode.")
    advanced.add_argument('--strand', type=str, choices=[x.value for x in Strand], default='both', help="On which strand to predict genes (default: %(default)s).")
    advanced.add_argument('--contigs-filter', type=str, help="Run only on selected contigs (comma separated string).")
    advanced.add_argument('--write-raw-scores', action='store_true', help="Instead of running gene calling, output the raw model scores as a .seg file.")
    advanced.add_argument('--max-transcripts', type=int, default=5, help="Maximum number of transcripts per gene (default: %(default)s).")
    advanced.add_argument('--allow-opposite-strand-overlaps', choices=['true', 'false'], default='true', help="Predict overlapping genes on opposite strands (default: %(default)s).")
    advanced.add_argument('--min-gene-score', type=str, default='dynamic', help="Minimum gene score for gene reporting. Can be a float value or 'dynamic' (default: %(default)s). Dynamic mode requires >=100,000 bp total input.")
    advanced.add_argument('--min-exon-size', type=int, default=1, help="Minimum exon size (default: %(default)s).")
    advanced.add_argument('--max-exon-size', type=int, default=10000, help="Maximum exon size (default: %(default)s).")
    advanced.add_argument('--min-intron-size', type=int, default=10, help="Minimum intron size (default: %(default)s).")
    advanced.add_argument('--max-intron-size', type=int, default=400, help="Maximum intron size (default: %(default)s).")
    advanced.add_argument('--cds-start-min-score', type=float, default=0.01, help="Minimum model score for considering a CDS start (default: %(default)s).")
    advanced.add_argument('--cds-end-min-score', type=float, default=0.01, help="Minimum model score for considering a CDS end (default: %(default)s).")
    advanced.add_argument('--exon-start-min-score', type=float, default=0.01, help="Minimum model score for considering an exon start (default: %(default)s).")
    advanced.add_argument('--exon-end-min-score', type=float, default=0.01, help="Minimum model score for considering an exon end (default: %(default)s).")
    advanced.add_argument('--gene-candidates', type=int, default=5000, help="Maximum number of gene candidates to consider (default: %(default)s).")

    args = parser.parse_args(argv)
    check_args(parser, args)

    return args


def main():
    args = parse_args()
    params = build_params_namedtuple(args)
    logfile = ''.join([params.basepath, '.log'])
    setup_logger(logfile, debug = params.debug, verbose= params.verbose)
    write_setup_info(params)
    process_genome(params)


if __name__ == "__main__":
    main()
