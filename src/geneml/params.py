import json
import os
from argparse import Namespace
from collections import namedtuple
from enum import Enum
from typing import Any


class Strand(Enum):
    """Supported prediction strand modes."""

    FORWARD = "forward"
    REVERSE = "reverse"
    BOTH = "both"


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that supports enums and namedtuples."""

    def default(self, obj) -> Any:
        """Serialize enums and namedtuples to JSON-compatible objects.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable representation of obj.
        """
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "_asdict"):  # for namedtuples
            return obj._asdict()
        return super().default(obj)


class Params(namedtuple('Params', (
    'min_intron_size', 'max_intron_size', 'cds_start_min_score', 'cds_end_min_score',
    'exon_start_min_score', 'exon_end_min_score', 'min_gene_score', 'gene_candidates',
    'model_path', 'context_length', 'strand',
    'contigs_filter', 'output_segs', 'output_genes', 'output_proteins',
    'num_cores', 'debug', 'verbose', 'basepath', 'inpath', 'outpath',
    'single_recurse_max_num_ops', 'recurse_region_max_num_ops',
    'max_transcripts', 'allow_opposite_strand_overlaps',
    'min_exon_size', 'max_exon_size', 'dynamic_scoring',
    'cpu_only',
))):
    """Immutable runtime parameter container used throughout geneML."""

    def to_json(self, **kwargs) -> str:
        """Serialize parameters to JSON.

        Args:
            **kwargs: Extra arguments passed to json.dumps.

        Returns:
            JSON string representation of the parameters.
        """
        return json.dumps(self._asdict(), cls=EnhancedJSONEncoder, **kwargs)

    def to_log_json(self, **kwargs) -> str:
        """Serialize parameters to grouped JSON for log output.

        Args:
            **kwargs: Extra arguments passed to json.dumps.

        Returns:
            JSON string representation grouped by parameter category.
        """
        grouped_params = {
            'paths': {
                'inpath': self.inpath,
                'outpath': self.outpath,
                'basepath': self.basepath,
                'contigs_filter': self.contigs_filter,
            },
            'model': {
                'model_path': self.model_path,
                'context_length': self.context_length,
            },
            'gene_calling': {
                'strand': self.strand,
                'max_transcripts': self.max_transcripts,
                'allow_opposite_strand_overlaps': (
                    self.allow_opposite_strand_overlaps
                ),
                'gene_candidates': self.gene_candidates,
            },
            'thresholds': {
                'dynamic_scoring': self.dynamic_scoring,
                'min_gene_score': self.min_gene_score,
                'min_exon_size': self.min_exon_size,
                'max_exon_size': self.max_exon_size,
                'min_intron_size': self.min_intron_size,
                'max_intron_size': self.max_intron_size,
                'cds_start_min_score': self.cds_start_min_score,
                'cds_end_min_score': self.cds_end_min_score,
                'exon_start_min_score': self.exon_start_min_score,
                'exon_end_min_score': self.exon_end_min_score,
            },
            'outputs': {
                'output_segs': self.output_segs,
                'output_genes': self.output_genes,
                'output_proteins': self.output_proteins,
            },
            'runtime': {
                'num_cores': self.num_cores,
                'cpu_only': self.cpu_only,
            },
            'logging': {
                'debug': self.debug,
                'verbose': self.verbose,
            },
            'internal_limits': {
                'single_recurse_max_num_ops': self.single_recurse_max_num_ops,
                'recurse_region_max_num_ops': self.recurse_region_max_num_ops,
            },
        }
        return json.dumps(grouped_params, cls=EnhancedJSONEncoder, **kwargs)


def get_basepath(inpath, outpath) -> str:
    """Resolve the base output path without extension.

    Args:
        inpath: Input sequence path.
        outpath: Optional explicit output path.

    Returns:
        Output base path without file extension.
    """
    if outpath:
        return os.path.splitext(outpath)[0]
    return os.path.splitext(inpath)[0]


def build_params_namedtuple(args: Namespace) -> Params:
    """Convert parsed CLI args to a Params namedtuple.

    numba cannot handle argparse Namespace or mixed-type dictionaries, but it can
    handle a namedtuple with stable field order.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Params instance used by downstream prediction code.
    """
    basepath = get_basepath(args.sequence, args.output)

    # Parse min_gene_score - can be 'dynamic' or a float value
    if args.min_gene_score.lower() == 'dynamic':
        dynamic_scoring = True
        min_gene_score = 0.5  # Fallback value used if dynamic scoring fails
    else:
        dynamic_scoring = False
        try:
            min_gene_score = float(args.min_gene_score)
        except ValueError as exc:
            raise ValueError(
                f"--min-gene-score must be either 'dynamic' or a valid float, "
                f"got: {args.min_gene_score}"
            ) from exc

    params_dict = {
        'model_path': args.model,
        'context_length': args.context_length if args.context_length else 800,
        'strand': Strand(args.strand),

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

        'max_transcripts': args.max_transcripts,
        'allow_opposite_strand_overlaps': args.allow_opposite_strand_overlaps == 'true',
        'min_exon_size': args.min_exon_size,
        'max_exon_size': args.max_exon_size,
        'min_intron_size': args.min_intron_size,
        'max_intron_size': args.max_intron_size,
        'cds_start_min_score': args.cds_start_min_score,
        'cds_end_min_score': args.cds_end_min_score,
        'exon_start_min_score': args.exon_start_min_score,
        'exon_end_min_score': args.exon_end_min_score,
        'min_gene_score': min_gene_score,
        'gene_candidates': args.gene_candidates,
        'dynamic_scoring': dynamic_scoring,

        'single_recurse_max_num_ops': 100000,
        'recurse_region_max_num_ops': 200000,
        'cpu_only': args.cpu_only,
    }

    return Params(*[params_dict[name] for name in Params._fields])
