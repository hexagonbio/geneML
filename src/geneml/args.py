import argparse

from geneml import __version__
from geneml.params import Strand


def positive_int(value) -> int:
    """Validate that input is a positive integer.

    Args:
        value: Input value to validate.

    Returns:
        The validated integer value.
    """
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, got {value}.")
    return ivalue


def unit_float(value) -> float:
    """Validate that input is a float in the range [0, 1].

    Args:
        value: Input value to validate.

    Returns:
        The validated float value.
    """
    fvalue = float(value)
    if not (0.0 <= fvalue <= 1.0):
        raise argparse.ArgumentTypeError(f"Must be between 0 and 1, got {value}.")
    return fvalue


def gene_score(value) -> str | float:
    """Validate gene score as 'dynamic' or a float in the range [0, 1].

    Args:
        value: Input value to validate.

    Returns:
        The string 'dynamic' or a validated float value.
    """
    if value == 'dynamic':
        return value
    try:
        return unit_float(value)
    except argparse.ArgumentTypeError as e:
        raise argparse.ArgumentTypeError(
            f"Must be a float between 0 and 1, or 'dynamic', got {value}."
            ) from e


def check_args(parser, args):
    if args.model and not args.context_length:
        parser.error("--context-length is required when using a custom model.")
    if args.min_exon_size > args.max_exon_size:
        parser.error(f"--min-exon-size ({args.min_exon_size}) cannot exceed "
                     f"--max-exon-size ({args.max_exon_size}).")
    if args.min_intron_size > args.max_intron_size:
        parser.error(f"--min-intron-size ({args.min_intron_size}) cannot exceed "
                     f"--max-intron-size ({args.max_intron_size}).")


def parse_args(argv=None):
    version_string = f"geneML {__version__}"
    parser = argparse.ArgumentParser(description=version_string, add_help=False)

    parser.add_argument('-h', '--help',
                        action='help',
                        help="Show this help message and exit.")
    parser.add_argument('--version',
                        action='version',
                        version=version_string,
                        help="Show version number and exit.")
    parser.add_argument('sequence',
                        type=str,
                        help="Sequence file in FASTA/GenBank/EMBL format.")
    parser.add_argument('-o', '--output',
                        type=str,
                        help="Gene annotations output path (default: based on input filename).")
    parser.add_argument('-g', '--genes',
                        type=str,
                        help="Gene sequences output path (default: None).")
    parser.add_argument('-p', '--proteins',
                        type=str,
                        help="Protein sequences output path (default: None).")
    parser.add_argument('-m', '--model',
                        type=str,
                        help="Path to model file (default: models/geneML_default.keras).")
    parser.add_argument('-cl', '--context-length',
                        type=positive_int,
                        help="Context length of the model.")
    parser.add_argument('-c', '--cores',
                        type=positive_int,
                        help="Number of cores to use for processing (default: all available).")

    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument('-v', '--verbose',
                          action='store_true',
                          help="Enable verbose mode.")
    advanced.add_argument('-d', '--debug',
                          action='store_true',
                          help="Enable debug mode.")
    advanced.add_argument('--cpu-only',
                          action='store_true',
                          help="Use CPU only for inference, disable GPU usage.")
    advanced.add_argument('--strand',
                          type=str,
                          choices=[x.value for x in Strand],
                          default='both',
                          help="On which strand to predict genes (default: %(default)s).")
    advanced.add_argument('--contigs-filter',
                          type=str,
                          help="Run only on selected contigs (comma separated string).")
    advanced.add_argument('--write-raw-scores',
                          action='store_true',
                          help=("Instead of running gene calling, "
                                "output the raw model scores as a .seg file."))
    advanced.add_argument('--max-transcripts',
                          type=positive_int,
                          default=5,
                          help="Maximum number of transcripts per gene (default: %(default)s).")
    advanced.add_argument('--allow-opposite-strand-overlaps',
                          choices=['true', 'false'],
                          default='true',
                          help=("Predict overlapping genes on opposite strands "
                                "(default: %(default)s)."))
    advanced.add_argument('--min-gene-score',
                          type=gene_score,
                          default='dynamic',
                          help=("Minimum gene score for gene reporting. "
                                "Can be a float value or 'dynamic' (default: %(default)s). "
                                "Dynamic mode requires >=100,000 bp total input."))
    advanced.add_argument('--min-exon-size',
                          type=positive_int,
                          default=1,
                          help="Minimum exon size (default: %(default)s).")
    advanced.add_argument('--max-exon-size',
                          type=positive_int,
                          default=30000,
                          help="Maximum exon size (default: %(default)s).")
    advanced.add_argument('--min-intron-size',
                          type=positive_int,
                          default=10,
                          help="Minimum intron size (default: %(default)s).")
    advanced.add_argument('--max-intron-size',
                          type=positive_int,
                          default=400,
                          help="Maximum intron size (default: %(default)s).")
    advanced.add_argument('--cds-start-min-score',
                          type=unit_float,
                          default=0.01,
                          help=("Minimum model score for considering a CDS start "
                                "(default: %(default)s)."))
    advanced.add_argument('--cds-end-min-score',
                          type=unit_float,
                          default=0.01,
                          help=("Minimum model score for considering a CDS end "
                                "(default: %(default)s)."))
    advanced.add_argument('--exon-start-min-score',
                          type=unit_float,
                          default=0.01,
                          help=("Minimum model score for considering an exon start "
                                "(default: %(default)s)."))
    advanced.add_argument('--exon-end-min-score',
                          type=unit_float,
                          default=0.01,
                          help=("Minimum model score for considering an exon end "
                                "(default: %(default)s)."))
    advanced.add_argument('--gene-candidates',
                          type=positive_int,
                          default=5000,
                          help=("Maximum number of gene candidates to consider per locus "
                                "(default: %(default)s)."))

    args = parser.parse_args(argv)
    check_args(parser, args)

    return args
