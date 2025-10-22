import json
import os
import numpy as np
from argparse import Namespace
from collections import namedtuple
from enum import Enum

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "_asdict"):  # for namedtuples
            return obj._asdict()
        return super().default(obj)


class Params(namedtuple('Params', (
    'min_intron_size', 'max_intron_size', 'cds_start_min_score', 'cds_end_min_score',
    'exon_start_min_score', 'exon_end_min_score', 'gene_candidates',
    'model_path', 'context_length', 'forward_strand_only', 'gene_range_time_limit',
    'contigs_filter', 'output_segs', 'output_genes', 'output_proteins',
    'num_cores', 'debug', 'verbose', 'basepath', 'inpath', 'outpath',
    'hardmask_repeats_min_size',
))):
    def to_json(self, **kwargs):
        return json.dumps(self._asdict(), cls=EnhancedJSONEncoder, **kwargs)


def get_basepath(inpath, outpath):
    if outpath:
        return os.path.splitext(outpath)[0]
    return os.path.splitext(inpath)[0]


def build_params_namedtuple(args: Namespace) -> Params:
    """
    numba can't handle Namespace, or a dictionary of mixed typed values, but it can handle a namedtuple.
    """
    basepath = get_basepath(args.sequence, args.output)
    params_dict = {
        'model_path': args.model,
        'context_length': args.context_length,
        'forward_strand_only': False,
        'gene_range_time_limit': np.inf if args.gene_range_time_limit is None else args.gene_range_time_limit,

        'contigs_filter': args.contigs_filter.split(',') if args.contigs_filter else None,
        'output_segs': args.write_raw_scores,
        'output_genes': args.genes,
        'output_proteins': args.proteins,

        'num_cores': args.cores,
        'debug': args.debug,
        'verbose': args.verbose,
        'basepath': basepath,
        'inpath': args.sequence,
        'outpath': args.output if args.output else ''.join([basepath, '.gff3']),

        'min_intron_size': args.min_intron_size,
        'max_intron_size': args.max_intron_size,
        'cds_start_min_score': args.cds_start_min_score,
        'cds_end_min_score': args.cds_end_min_score,
        'exon_start_min_score': args.exon_start_min_score,
        'exon_end_min_score': args.exon_end_min_score,
        'gene_candidates': args.gene_candidates,
        'hardmask_repeats_min_size': args.hardmask_repeats_min_size,
    }

    return Params(*[params_dict[name] for name in Params._fields])
