"""Functions for parsing and validating input FASTA sequences."""

import logging
import re

from Bio.Data import IUPACData
from helperlibs.bio import seqio

AMBIGUOUS_DNA_LETTERS = frozenset(IUPACData.ambiguous_dna_letters) - frozenset('ACGTN')
AMBIGUOUS_DNA_TO_N = str.maketrans({base: 'N' for base in AMBIGUOUS_DNA_LETTERS})

logger = logging.getLogger("geneml")


def hardmask_long_repeats(seq: str, min_repeat_len: int) -> str:
    """Mask repeats of at least min_repeat_len bases with N."""
    if min_repeat_len <= 0:
        return seq

    def mask(match: re.Match[str]) -> str:
        return 'N' * len(match.group())

    return re.sub(
        rf'[a-z]{{{min_repeat_len},}}',
        mask,
        seq,
    )


def parse_contigs(inpath: str, contigs_filter: list[str] | None,
                  mask_repeats: bool = False, mask_repeats_min_len: int = 0,
                  ) -> tuple[dict[str, str], int]:
    """Parse and validate contig sequences from an input file.

    Reads genome records, restricted to IDs in contigs_filter if provided.
    Sequences are converted to uppercase, ambiguous IUPAC DNA codes are mapped to N,
    and the result is validated to contain only valid nucleotide characters.

    Args:
        inpath: Path to an input sequence file in FASTA/GenBank/EMBL format
        contigs_filter: Optional list of contig IDs to include
        mask_repeats: If True, mask repetitive regions (lowercase letters) with N.
        mask_repeats_min_len: Minimum length of repetitive regions to mask.

    Returns:
        Tuple of (contigs, genome_size) where contigs is a dictionary mapping contig ID to
        sequence and genome_size is the total number of bases
    """
    contigs = {}
    genome_size = 0
    to_process = set(contigs_filter or [])

    for record in seqio.parse(inpath):
        if contigs_filter and record.id not in to_process:
            # Skip this record
            continue
        to_process.discard(record.id)

        seq = str(record.seq)

        if mask_repeats and mask_repeats_min_len > 0:
            seq = hardmask_long_repeats(seq, mask_repeats_min_len)

        seq = seq.upper()

        ambiguous_dna_letters = sorted(set(seq) & AMBIGUOUS_DNA_LETTERS)
        if ambiguous_dna_letters:
            logger.warning(
                'Contig %s contains ambiguous DNA codes %s; converting them to N.',
                record.id,
                ', '.join(ambiguous_dna_letters),
            )

        seq = seq.translate(AMBIGUOUS_DNA_TO_N)
        # Check if sequence is valid
        if not seq:
            raise ValueError(f"Contig {record.id} has no sequence.")
        invalid_chars = set(seq) - set('ACGTN')
        if invalid_chars:
            raise ValueError(
                f"Sequence of contig {record.id} contains invalid characters: "
                f"{', '.join(sorted(invalid_chars))}."
            )

        contigs[record.id] = seq
        genome_size += len(seq)

    if to_process:
        raise ValueError(
            f"The following contig IDs were specified with --contigs-filter "
            f"but not found in input file: {', '.join(sorted(to_process))}"
        )

    return contigs, genome_size
