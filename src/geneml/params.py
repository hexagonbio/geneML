from argparse import Namespace
from collections import namedtuple

Params = namedtuple('Params', (
    'min_intron_size', 'max_intron_size', 'cds_start_min_score', 'cds_end_min_score',
    'exon_start_min_score', 'exon_end_min_score', 'num_candidate_gene_calls_per_region',
    'model_path', 'forward_strand_only', 'gene_range_time_limit',
    'contigs_filter', 'output_segs',
    'num_cores', 'debug', 'input', 'output',
))

def build_params_namedtuple(args: Namespace) -> Params:
    """
    numba can't handle Namespace, or a dictionary of mixed typed values, but it can handle a namedtuple.
    """

    if args.contigs_filter is not None:
        contigs_filter = args.contigs_filter.split(',')
        output_segs = True
    else:
        contigs_filter = None
        output_segs = False

    params_dict = {
        'model_path': args.model,
        'forward_strand_only': False,
        'gene_range_time_limit': 300,

        'contigs_filter': contigs_filter,
        'output_segs': output_segs,

        'num_cores': args.num_cores,
        'debug': args.debug,
        'input': args.input,
        'output': args.output,

        # default sensitivity parameters
        'min_intron_size': 30,
        'max_intron_size': 400,
        'cds_start_min_score': 0.01,
        'cds_end_min_score': 0.01,
        'exon_start_min_score': 0.01,
        'exon_end_min_score': 0.01,
        'num_candidate_gene_calls_per_region': 100,
    }

    if args.sensitive:
        params_dict.update({
            'min_intron_size': 30,
            'max_intron_size': 1000,
            'cds_start_min_score': 0.001,
            'cds_end_min_score': 0.001,
            'exon_start_min_score': 0.001,
            'exon_end_min_score': 0.001,
            'num_candidate_gene_calls_per_region': 1000,
        })

    return Params(*[params_dict[name] for name in Params._fields])
