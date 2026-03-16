from collections import namedtuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numba import typed, typeof

CDS_START = 0
CDS_END = 1
EXON_START = 2
EXON_END = 3

# Note: GeneEvent uses inclusive positions
GeneEvent = namedtuple('GeneEvent', ['pos', 'type', 'score'])

GeneEventNumbaType = typeof(GeneEvent(1, CDS_START, np.float32(0.5)))
GeneCallNumbaType = typeof(typed.List.empty_list(GeneEventNumbaType))


class TranscriptVariant(Enum):
    """Classification labels describing transcript isoform relationships."""

    UNKNOWN = 0
    PRIMARY = 1
    INTRON_RETENTION = 2
    EXON_SKIPPING = 3
    ALT_FIRST_EXON = 4
    ALT_LAST_EXON = 5
    ALT_5_SPLICE_SITE = 6
    ALT_3_SPLICE_SITE = 7
    COMPLEX = 8


@dataclass
class Exon:
    """Coding exon with genomic bounds, defining events, and score metadata."""

    start: int
    end: int
    events: tuple[GeneEvent, ...]
    score: float
    phase: int

def __post_init__(self) -> None:
    """Validate that each exon has exactly one start and one end event."""
    if not self.events or len(self.events) != 2:
        raise ValueError('An exon must have exactly two gene events (start and end).')


@dataclass
class Transcript:
    """Transcript model containing ordered exons and variant metadata."""

    start: int
    end: int
    strand: int
    events: tuple[GeneEvent, ...]
    score: float
    exons: tuple[Exon, ...]
    group_id: int
    transcript_id: str = ""
    transcript_variant: TranscriptVariant = TranscriptVariant.UNKNOWN

    def __post_init__(self) -> None:
        """Validate transcript exon content and coordinate consistency."""
        if not self.exons:
            raise ValueError('A transcript must have at least one exon.')

        for exon in self.exons:
            if exon.start < self.start or exon.end > self.end:
                raise ValueError(f'Exon ({exon.start}, {exon.end}) '
                                 f'is out of transcript bounds ({self.start}, {self.end}).')

    def set_transcript_id(self, transcript_id: str) -> None:
        """Assign a stable transcript identifier.

        Args:
            transcript_id: Identifier to assign.

        Returns:
            None.
        """
        self.transcript_id = transcript_id

    def set_transcript_variant(self, transcript_variant: TranscriptVariant) -> None:
        """Assign transcript variant classification.

        Args:
            transcript_variant: Variant label to assign.

        Returns:
            None.
        """
        self.transcript_variant = transcript_variant

    def classify_transcript_variant(self, primary_transcript: 'Transcript') -> TranscriptVariant:
        """Classify this transcript relative to the primary isoform.

        Args:
            primary_transcript: Primary transcript used as reference.

        Returns:
            Transcript variant label for this transcript.
        """
        assert primary_transcript.transcript_variant == TranscriptVariant.PRIMARY

        if self.exons == primary_transcript.exons:
            return TranscriptVariant.PRIMARY

        from geneml.splicing import get_alternative_transcript_variant
        return get_alternative_transcript_variant(primary_transcript, self)

    def overlaps_with(self, other: 'Transcript', ignore_strand: bool = False) -> bool:
        """Check whether two transcript intervals overlap.

        Args:
            other: Transcript to compare against.
            ignore_strand: Whether strand must match.

        Returns:
            True when intervals overlap under the chosen strand rule.
        """
        # by default only consider overlaps on the same strand
        if not ignore_strand and self.strand != other.strand:
            return False
        return self.start < other.end and other.start < self.end


@dataclass
class Gene:
    """Gene model containing transcript isoforms and locus coordinates."""

    gene_id: str
    start: int
    end: int
    strand: int
    transcripts: tuple[Transcript, ...]
    score: float

    def __post_init__(self) -> None:
        """Validate transcripts and assign IDs and variant labels."""
        if not self.transcripts:
            raise ValueError('A gene must have at least one transcript.')

        for i, transcript in enumerate(self.transcripts):
            transcript_id = f'{self.gene_id}_mRNA{i+1}'
            transcript.set_transcript_id(transcript_id)

            if transcript.start < self.start or transcript.end > self.end:
                raise ValueError(f'Transcript {transcript} is out of gene bounds: {self.start}, {self.end}.')

            # First transcript always denotes the primary splicing type
            if i == 0:
                transcript.set_transcript_variant(TranscriptVariant.PRIMARY)
                primary = transcript
            else:
                transcript_variant = transcript.classify_transcript_variant(primary)
                transcript.set_transcript_variant(transcript_variant)
