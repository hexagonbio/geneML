import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from Bio import SeqIO
from tqdm import tqdm

from gene_ml.gene_caller import build_gene_calls, GeneEvent, EXON_END, CDS_END, run_model
from gene_ml.model_loader import ExonIntron6ClassModel
from gene_ml.outputs import build_gff_coords


def process_contig(contig_id: str, preds: dict, rc_preds: dict, seq: str, rc_seq: str,
                   debug=False) -> tuple[str, list[list[float | GeneEvent | bool]]]:
    """
    Returns a python-only data structure so it can be pickled for either joblib or crossing over process boundaries
    """
    filtered_scored_gene_calls, seen, preds, rc_preds = build_gene_calls(preds, rc_preds, seq, rc_seq, debug=debug)
    rebuilt_results = []
    for score, gene_call, is_rc in filtered_scored_gene_calls:
        rebuilt_gene_call = []
        for i, event in enumerate(gene_call):
            # TODO: does this gene call fixing belong elsewhere?
            if i < len(gene_call) - 1 and gene_call[i].type == EXON_END and gene_call[i + 1].type == CDS_END:
                continue  # skip exon end if followed by cds end
            rebuilt_gene_call.append(GeneEvent(pos=event.pos, type=event.type, score=event.score))
        rebuilt_results.append([score, rebuilt_gene_call, is_rc])
    return contig_id, rebuilt_results


def process_genome(path, outpath, num_cores=1, contigs_filter=None, debug=False, model_path=None):

    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    contigs = {}
    for record in SeqIO.parse(path, "fasta"):
        if contigs_filter is not None and record.id not in contigs_filter:
            continue
        contigs[record.id] = str(record.seq).upper()

    contigs_by_size = sorted(contigs.items(), key=lambda x: len(x[1]), reverse=False)
    split_point = len(contigs_by_size) // 3
    contigs_by_size = contigs_by_size[:split_point] + contigs_by_size[split_point:][::-1]

    model = ExonIntron6ClassModel(path=model_path)

    if num_cores == 1:
        import time
        print('Running in single-threaded mode')
        results = {}
        for contig_id, seq in contigs_by_size:
            print(f'Processing contig {contig_id} of size {len(seq)}')
            start_time = time.time()
            preds, rc_preds, seq, rc_seq = run_model(model, seq)
            _, r = process_contig(contig_id, preds, rc_preds, seq, rc_seq, debug=debug)
            results[contig_id] = r
            end_time = time.time()
            elapsed = end_time - start_time
            print(f'Finished processing contig {contig_id} in {end_time - start_time:.2f} seconds, {len(seq)/elapsed:.2f} bp/s')
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            future_to_args = {}
            with tqdm(total=len(contigs_by_size)) as progress:
                for contig_id, seq in contigs_by_size:
                    progress.set_description(f'Model inference for {contig_id} ({len(seq)} bp)')
                    progress.update()
                    preds, rc_preds, seq, rc_seq = run_model(model, seq)
                    future = pool.submit(process_contig, contig_id, preds, rc_preds, seq, rc_seq)
                    future_to_args[future] = contig_id

            results = {}
            with tqdm(total=len(future_to_args)) as progress:
                for future in as_completed(future_to_args):
                    contig_id = future_to_args[future]
                    progress.set_description(f'Gene reporting for {contig_id}', refresh=False)
                    progress.update()

                    _, r = future.result()
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

    with open(outpath, 'w') as f:
        for gff_row in all_gff_rows:
            formatted_gff_row = '\t'.join(map(str, gff_row))
            f.write(f'{formatted_gff_row}\n')


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
