from concurrent.futures import ProcessPoolExecutor, as_completed

import gc
import os
import time
from Bio import SeqIO
from tqdm import tqdm

from geneml.gene_caller import CDS_END, EXON_END, GeneEvent, build_gene_calls, run_model
from geneml.model_loader import get_cached_gene_ml_model
from geneml.outputs import build_gff_coords, build_prediction_scores_seg
from geneml.utils import compute_optimal_num_parallelism


def process_contig(contig_id: str, seq: str, model_path: str, tensorflow_thread_count=None, output_segs=False,
                   debug=False) -> tuple[str, list[list[float | GeneEvent | bool]], list[str], str]:
    """
    Returns a python-only data structure so it can be pickled for either joblib or crossing over process boundaries
    """
    import silence_tensorflow.auto
    silence_tensorflow.auto
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(tensorflow_thread_count)
    tf.config.threading.set_intra_op_parallelism_threads(tensorflow_thread_count)

    model = get_cached_gene_ml_model(model_path)
    preds, rc_preds, seq, rc_seq = run_model(model, seq)
    filtered_scored_gene_calls, logs = build_gene_calls(preds, rc_preds, seq, rc_seq, contig_id, debug=debug)

    rebuilt_results = []
    for score, gene_call, is_rc in filtered_scored_gene_calls:
        rebuilt_gene_call = []
        for i, event in enumerate(gene_call):
            # TODO: does this gene call fixing belong elsewhere?
            if i < len(gene_call) - 1 and gene_call[i].type == EXON_END and gene_call[i + 1].type == CDS_END:
                continue  # skip exon end if followed by cds end
            rebuilt_gene_call.append(GeneEvent(pos=event.pos, type=event.type, score=event.score))
        rebuilt_results.append([score, rebuilt_gene_call, is_rc])
    rebuilt_logs = [str(log) for log in logs]

    if output_segs:
        segs = str(build_prediction_scores_seg(contig_id, preds, rc_preds))
    else:
        segs = ''

    # explicitly clean up memory after finishing a contig
    del preds
    del rc_preds
    del filtered_scored_gene_calls
    del logs
    gc.collect()

    return contig_id, rebuilt_results, rebuilt_logs, segs


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


def process_genome(path, outpath, num_cores=1, contigs_filter=None, debug=False, model_path=None):
    all_logs = []
    all_logs.append(f'Processing {path} with {num_cores} cores, model_path={model_path}, contigs_filter={contigs_filter}')
    genome_start_time = time.time()

    contigs = {}
    genome_size = 0
    for record in SeqIO.parse(path, "fasta"):
        if contigs_filter is not None and record.id not in contigs_filter:
            continue
        contigs[record.id] = str(record.seq).upper()
        genome_size += len(record.seq)

    if num_cores is None:
        num_cores, tensorflow_thread_count = compute_optimal_num_parallelism(num_contigs=len(contigs))
        log = f'Based on available memory, setting parallelism to {num_cores} parallel processes and tensorflow threads to {tensorflow_thread_count}'
        all_logs.append(log)
        print(log)
    else:
        tensorflow_thread_count = None

    reordered_contigs = reorder_contigs(contigs, num_cores)

    results = {}
    all_segs = []
    if num_cores == 1:
        print('Running from main thread, parallelism only for tensorflow')
        for contig_id, seq in reordered_contigs:
            print(f'Processing contig {contig_id} of size {len(seq)}')
            start_time = time.time()
            _, r, logs, segs = process_contig(contig_id, seq, model_path, tensorflow_thread_count, bool(contigs_filter), debug=debug)
            results[contig_id] = r
            all_logs.extend(logs)
            all_segs.append(segs)
            elapsed = time.time() - start_time
            print(f'Finished processing contig {contig_id} in {elapsed:.2f} seconds, {len(seq)/elapsed:.2f} bp/s')
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            future_to_args = {}
            for contig_id, seq in reordered_contigs:
                future = pool.submit(process_contig, contig_id, seq, model_path, tensorflow_thread_count)
                future_to_args[future] = contig_id, len(seq)

            with tqdm(total=genome_size, unit='bp', smoothing=0.1, unit_scale=True, mininterval=1) as progress:
                progress.set_description(f'Processing {path}')
                for future in as_completed(future_to_args):
                    contig_id, seq_len = future_to_args[future]
                    progress.update(seq_len)

                    _, r, logs, _ = future.result()
                    all_logs.extend(logs)
                    results[contig_id] = r

    print('Finished processing all contigs')
    gene_count = 0
    all_gff_rows = []
    for contig_id, seq in contigs.items():
        if contig_id not in results:
            continue
        filtered_scored_gene_calls = results[contig_id]
        gff_rows = []
        for score, gene_call, is_rc in filtered_scored_gene_calls:
            gene_count += 1
            gff_rows.extend(build_gff_coords(contig_id, 'geneML', f'GML{gene_count}',
                                             gene_call, 0, len(seq), is_rc))
        all_gff_rows.extend(sorted(gff_rows, key=lambda o: o[3]))

    if dirname := os.path.dirname(outpath):
        os.makedirs(dirname, exist_ok=True)
    with open(outpath, 'w') as f:
        for gff_row in all_gff_rows:
            formatted_gff_row = '\t'.join(map(str, gff_row))
            f.write(f'{formatted_gff_row}\n')

    elapsed = time.time() - genome_start_time
    log = f'Finished processing {path}, {genome_size/1e6:.1f}MB, in {elapsed/60:.2f} minutes'
    print(log)
    all_logs.append(log)

    with open(outpath + '.log', 'w') as f:
        for log in all_logs:
            f.write(f'{log}\n')

    if all_segs:
        with open(outpath.replace('.gff', '') + '.seg', 'w') as f:
            f.write('#track graphType=heatmap maxHeightPixels=20:20:20 color=0,0,255 altColor=255,0,0\n')
            for segs in all_segs:
                f.write(f'{segs}\n')


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process a genome file and output GFF coordinates.")
    parser.add_argument("input", type=str, help="Path to the input genome file.")
    parser.add_argument("output", type=str, help="Path to the output GFF file.")
    parser.add_argument('--contigs-filter', type=str, default=None,)
    parser.add_argument('--num-cores', type=int, default=None, help="Number of cores to use for processing (default: all).")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('--model', type=str, default=None, help="model id or path to model file")
    args = parser.parse_args()

    if args.contigs_filter is not None:
        contigs_filter = args.contigs_filter.split(',')
    else:
        contigs_filter = None

    process_genome(args.input, args.output, contigs_filter=contigs_filter, num_cores=args.num_cores, debug=args.debug,
                   model_path=args.model)


if __name__ == "__main__":
    main()
