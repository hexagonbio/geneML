from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from numba import typed, typeof

CDS_START = 0
CDS_END = 1
EXON_START = 2
EXON_END = 3

GeneEvent = namedtuple('GeneEvent', ['pos', 'type', 'score'])

GeneEventNumbaType = typeof(GeneEvent(1, CDS_START, np.float32(0.5)))
GeneCallNumbaType = typeof(typed.List.empty_list(GeneEventNumbaType))


@dataclass
class Exon:
    start: int
    end: int
    events: tuple[GeneEvent, ...]

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
    transcript_id: str = ""

    def __post_init__(self):
        if not self.exons:
            raise ValueError('A transcript must have at least one exon.')

    def set_transcript_id(self, transcript_id: str):
        self.transcript_id = transcript_id

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

    def __post_init__(self):
        if not self.transcripts:
            raise ValueError('A gene must have at least one transcript.')

        for transcript in self.transcripts:
            transcript_id = f'{self.gene_id}_mRNA'
            transcript.set_transcript_id(transcript_id)

            if transcript.start < self.start or transcript.end > self.end:
                raise ValueError(f'Transcript {transcript.transcript_id} is out of gene bounds.')
