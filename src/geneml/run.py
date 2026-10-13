"""Run the geneML pipeline."""

import logging
import time

import enlighten
from helperlibs.bio import seqio

from geneml.inputs import parse_contigs
from geneml.outputs import build_cds_sequences, write_fasta, write_gff_file, write_seg_file
from geneml.params import Params
from geneml.pipeline import process_genome

logger = logging.getLogger("geneml")


def run_geneML(params: Params) -> None:
    start_time = time.time()

    contigs, genome_size = parse_contigs(params.inpath, params.contigs_filter,
                            params.mask_repeats, params.mask_repeats_min_len)

    manager = enlighten.get_manager()
    progress = manager.counter(desc=f'Processing {params.inpath}', total=genome_size,
                               unit='bp', color='green')
    progress.refresh()   # draw empty progress bar
    try:
        result = process_genome(contigs, params,
                                on_contig_done=lambda contig_id, seq_len: progress.update(seq_len))
    finally:
        progress.close()
        manager.stop()

    if result.genes_by_contig_id:
        logger.info('Writing gene predictions to %s', params.outpath)
        write_gff_file(contigs, result.genes_by_contig_id, params.outpath,
                       result.mean_gene_score)

        if params.output_genes or params.output_proteins:
            original_contigs = {record.id: str(record.seq) for record in seqio.parse(params.inpath)}
            cdses_by_transcript = build_cds_sequences(original_contigs, result.genes_by_contig_id)

            if params.output_genes:
                logger.info('Writing gene sequences to %s', params.output_genes)
                write_fasta(cdses_by_transcript, params.output_genes, sequence_type = 'cds')

            if params.output_proteins:
                logger.info('Writing protein sequences to %s', params.output_proteins)
                write_fasta(cdses_by_transcript, params.output_proteins, sequence_type = 'protein')

    if result.segs:
        write_seg_file(result.segs, params)

    elapsed = time.time() - start_time
    logger.info('Finished processing %s, %.1fMB, in %.2f minutes',
                params.inpath, genome_size/1e6, elapsed/60)
