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


class SplicingType(Enum):
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
    start: int
    end: int
    events: tuple[GeneEvent, ...]
    score: float
    phase: int

def __post_init__(self):
    if not self.events or len(self.events) != 2:
        raise ValueError('An exon must have exactly two gene events (start and end).')


@dataclass
class Transcript:
    start: int
    end: int
    strand: int
    events: tuple[GeneEvent, ...]
    score: float
    exons: tuple[Exon, ...]
    group_id: int = -1
    transcript_id: str = ""
    splicing_type: SplicingType = SplicingType.UNKNOWN

    def __post_init__(self):
        if not self.exons:
            raise ValueError('A transcript must have at least one exon.')

        for exon in self.exons:
            if exon.start < self.start or exon.end > self.end:
                raise ValueError(f'Exon ({exon.start}, {exon.end}) '
                                 f'is out of transcript bounds ({self.start}, {self.end}).')

    def set_transcript_id(self, transcript_id: str):
        self.transcript_id = transcript_id

    def set_splicing_type(self, splicing_type: SplicingType):
        self.splicing_type = splicing_type

    def classify_splicing_type(self, primary_transcript: 'Transcript'):
        assert primary_transcript.splicing_type == SplicingType.PRIMARY

        if self.exons == primary_transcript.exons:
            return SplicingType.PRIMARY

        from geneml.splicing import get_alternative_splicing_type
        return get_alternative_splicing_type(primary_transcript, self)

    def overlaps_with(self, other: 'Transcript', ignore_strand: bool = False) -> bool:
        # by default only consider overlaps on the same strand
        if not ignore_strand and self.strand != other.strand:
            return False
        return self.start < other.end and other.start < self.end


@dataclass
class Gene:
    gene_id: str
    start: int
    end: int
    strand: int
    transcripts: tuple[Transcript, ...]
    score: float

    def __post_init__(self):
        if not self.transcripts:
            raise ValueError('A gene must have at least one transcript.')

        for i, transcript in enumerate(self.transcripts):
            transcript_id = f'{self.gene_id}_mRNA{i+1}'
            transcript.set_transcript_id(transcript_id)

            if transcript.start < self.start or transcript.end > self.end:
                raise ValueError(f'Transcript {transcript} is out of gene bounds: {self.start}, {self.end}.')

            # First transcript always denotes the primary splicing type
            if i == 0:
                transcript.set_splicing_type(SplicingType.PRIMARY)
                primary = transcript
            else:
                splicing_type = transcript.classify_splicing_type(primary)
                transcript.set_splicing_type(splicing_type)
