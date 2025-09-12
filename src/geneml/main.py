import argparse
import gc
import json
import logging
import os
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import tensorflow as tf
from helperlibs.bio import seqio
from tqdm import tqdm

from geneml import __version__
from geneml.gene_caller import CDS_END, EXON_END, GeneEvent, build_gene_calls, run_model
from geneml.model_loader import get_cached_gene_ml_model
from geneml.outputs import build_prediction_scores_seg, write_fasta, write_gff_file
from geneml.params import build_params_namedtuple
from geneml.utils import compute_optimal_num_parallelism

logger = logging.getLogger("geneml")

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

def process_contig(contig_id: str, seq: str, params: namedtuple, tensorflow_thread_count=None) -> tuple[str, list[list[float | GeneEvent | bool]], list[str], str]:
    """
    Returns a python-only data structure so it can be pickled for either joblib or crossing over process boundaries
    """
    tf.config.threading.set_inter_op_parallelism_threads(tensorflow_thread_count)
    tf.config.threading.set_intra_op_parallelism_threads(tensorflow_thread_count)

    #Get model scores
    model = get_cached_gene_ml_model(params.model_path, params.context_length)
    preds, rc_preds, seq, rc_seq = run_model(model, seq)

    if params.output_segs:
        segs = str(build_prediction_scores_seg(contig_id, preds, rc_preds))
        return contig_id, [], [], segs

    filtered_scored_gene_calls = build_gene_calls(preds, rc_preds, seq, rc_seq, contig_id, params)

    rebuilt_results = []
    for score, gene_call, is_rc in filtered_scored_gene_calls:
        rebuilt_gene_call = []
        for i, event in enumerate(gene_call):
            # TODO: does this gene call fixing belong elsewhere?
            if i < len(gene_call) - 1 and gene_call[i].type == EXON_END and gene_call[i + 1].type == CDS_END:
                continue  # skip exon end if followed by cds end
            rebuilt_gene_call.append(GeneEvent(pos=event.pos, type=event.type, score=event.score))
        rebuilt_results.append([score, rebuilt_gene_call, is_rc])

    # explicitly clean up memory after finishing a contig
    del preds
    del rc_preds
    del filtered_scored_gene_calls
    gc.collect()

    return contig_id, rebuilt_results, None


def reorder_contigs(contigs, num_cores):
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


def process_genome(params: namedtuple):
    num_cores = params.num_cores
    logger.info(f"Running geneML version {__version__}")
    parameter_info = '\n'.join(["Parameters:", json.dumps(params._asdict(), indent=2)])
    logger.info(parameter_info)
    genome_start_time = time.time()

    contigs = {}
    genome_size = 0
    for record in seqio.parse(params.inpath):
        if params.contigs_filter is not None and record.id not in params.contigs_filter:
            continue
        contigs[record.id] = str(record.seq).upper()
        genome_size += len(record.seq)

    if num_cores is None:
        num_cores, tensorflow_thread_count = compute_optimal_num_parallelism(num_contigs=len(contigs))
        logger.info(f'Based on available memory, setting parallelism to {num_cores} parallel processes and tensorflow threads to {tensorflow_thread_count or "all available"}')
    else:
        tensorflow_thread_count = None

    reordered_contigs = reorder_contigs(contigs, num_cores)

    results = {}
    all_segs = []
    if num_cores == 1:
        logger.info('Running from main thread, parallelism only for tensorflow')
        for contig_id, seq in reordered_contigs:
            logger.info(f'Processing contig {contig_id} of size {len(seq)}')
            start_time = time.time()
            _, r, segs = process_contig(contig_id, seq, params, tensorflow_thread_count)
            results[contig_id] = r
            if segs:
                all_segs.append(segs)
            elapsed = time.time() - start_time
            logger.info(f'Finished processing contig {contig_id} in {elapsed:.2f} seconds, {len(seq)/elapsed:.2f} bp/s')
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            future_to_args = {}
            for contig_id, seq in reordered_contigs:
                future = pool.submit(process_contig, contig_id, seq, params, tensorflow_thread_count)
                future_to_args[future] = contig_id, len(seq)

            with tqdm(total=genome_size, unit='bp', smoothing=0.1, unit_scale=True, mininterval=1) as progress:
                progress.set_description(f'Processing {params.inpath}')
                for future in as_completed(future_to_args):
                    contig_id, seq_len = future_to_args[future]
                    progress.update(seq_len)
                    results[contig_id] = r

    logger.info('Finished processing all contigs')

    if all_segs:
        with open(params.basepath+'.seg', 'w') as f:
            f.write('#track graphType=heatmap maxHeightPixels=20:20:20 color=0,0,255 altColor=255,0,0\n')
            for segs in all_segs:
                f.write(f'{segs}\n')
    else:
        write_gff_file(contigs, results, params.outpath)
        if params.output_genes:
            write_fasta(contigs, results, params.output_genes, sequence_type = 'cds')
        if params.output_proteins:
            write_fasta(contigs, results, params.output_proteins, sequence_type = 'fasta')

    elapsed = time.time() - genome_start_time
    logger.info(f'Finished processing {params.inpath}, {genome_size/1e6:.1f}MB, in {elapsed/60:.2f} minutes')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=f"geneML {__version__}")
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
    advanced.add_argument('--contigs-filter', type=str, help="Run only on selected contigs (comma separated string).")
    advanced.add_argument('--write-raw-scores', action='store_true', help="Instead of running gene calling, output the raw model scores as a .seg file.")
    advanced.add_argument('--min-intron-size', type=int, default=10, help="Minimum intron size (default: %(default)s).")
    advanced.add_argument('--max-intron-size', type=int, default=400, help="Maximum intron size (default: %(default)s).")
    advanced.add_argument('--cds-start-min-score', type=float, default=0.01, help="Minimum model score for considering a CDS start (default: %(default)s).")
    advanced.add_argument('--cds-end-min-score', type=float, default=0.01, help="Minimum model score for considering a CDS end (default: %(default)s).")
    advanced.add_argument('--exon-start-min-score', type=float, default=0.01, help="Minimum model score for considering an exon start (default: %(default)s).")
    advanced.add_argument('--exon-end-min-score', type=float, default=0.01, help="Minimum model score for considering an exon end (default: %(default)s).")
    advanced.add_argument('--gene-candidates', type=int, default=100, help="Maximum number of gene candidates to consider (default: %(default)s).")

    args = parser.parse_args(argv)
    check_args(parser, args)

    return args


def main():
    args = parse_args()
    params = build_params_namedtuple(args)
    logfile = ''.join([params.basepath, '.log'])
    setup_logger(logfile, debug = params.debug, verbose= params.verbose)
    process_genome(params)


if __name__ == "__main__":
    main()
