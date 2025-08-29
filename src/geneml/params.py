from argparse import Namespace
from collections import namedtuple

Params = namedtuple('Params', (
    'min_intron_size', 'max_intron_size', 'cds_start_min_score', 'cds_end_min_score',
    'exon_start_min_score', 'exon_end_min_score', 'gene_candidates',
    'model_path', 'forward_strand_only', 'gene_range_time_limit',
    'contigs_filter', 'output_segs', 'output_genes', 'output_proteins',
    'num_cores', 'debug', 'input', 'output',
))

def build_params_namedtuple(args: Namespace) -> Params:
    """
    numba can't handle Namespace, or a dictionary of mixed typed values, but it can handle a namedtuple.
    """
    params_dict = {
        'model_path': args.model,
        'forward_strand_only': False,
        'gene_range_time_limit': 300,

        'contigs_filter': args.contigs_filter.split(',') if args.contigs_filter else None,
        'output_segs': args.write_raw_scores,
        'output_genes': args.genes,
        'output_proteins': args.proteins,

        'num_cores': args.num_cores,
        'debug': args.debug,
        'input': args.input,
        'output': args.output,

        'min_intron_size': args.min_intron_size,
        'max_intron_size': args.max_intron_size,
        'cds_start_min_score': args.cds_start_min_score,
        'cds_end_min_score': args.cds_end_min_score,
        'exon_start_min_score': args.exon_start_min_score,
        'exon_end_min_score': args.exon_end_min_score,
        'gene_candidates': args.gene_candidates,
    }

    return Params(*[params_dict[name] for name in Params._fields])
