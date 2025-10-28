import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from geneml.gene_caller import get_gene_ml_events, produce_gene_calls
from geneml.model_loader import ResidualModelBase
from geneml.params import Params
from geneml.types import CDS_END, CDS_START, EXON_END, EXON_START, Exon, Gene, GeneEvent, Transcript

logger = logging.getLogger("geneml")


def run_model(model: ResidualModelBase, seq: str, chunk_size=100000, padding=1000):
    """
    Predicts the sequence in chunks of a given size to control memory usage, with some padding to handle
    the sequence context
    """
    seq_len = len(seq)
    pred_list = []
    for i in range(0, seq_len, chunk_size):
        start = i
        end = min(seq_len, i + chunk_size)
        padded_start = max(0, start - padding)
        padded_end = min(seq_len, end + padding)
        preds = model.predict(seq[padded_start:padded_end], return_dict=False)
        pred_list.append(preds[:, (start-padded_start):(end-padded_start)])
    return np.concatenate(pred_list, axis=1)


def filter_gene_events(gene_events: list[GeneEvent]) -> list[GeneEvent]:
    if not gene_events:
        return []
    filtered_events = []
    for i, event in enumerate(gene_events):
        # skip exon end if followed by cds end
        if (i < len(gene_events) - 1 and
            event.type == EXON_END and
            gene_events[i + 1].type == CDS_END):
            continue
        filtered_events.append(event)
    return filtered_events


def create_exons(gene_events: list[GeneEvent]) -> list[Exon]:
    if not gene_events:
        return []
    assert len(gene_events) % 2 == 0, 'There should be an even number of gene events.'
    exons = []
    for event, next_event in zip(gene_events[::2], gene_events[1::2]):
        if not (event.type in (CDS_START, EXON_START) and next_event.type in (CDS_END, EXON_END)):
            raise ValueError(f'Invalid gene events pair (expected start event and end event): {event}, {next_event}')
        exon = Exon(
            start=event.pos,
            end=next_event.pos + 1,  # end is exclusive
            events=(event, next_event)
        )
        exons.append(exon)
    return exons


def create_transcripts(scored_gene_calls: list[tuple[float, list[GeneEvent]]],
                       contig_length: int, is_rc: bool) -> list[Transcript]:
    transcripts = []
    strand = -1 if is_rc else 1
    for score, gene_events in scored_gene_calls:
        filtered_events = filter_gene_events(gene_events)
        if is_rc:
            start_pos = contig_length - filtered_events[-1].pos
            end_pos = contig_length - filtered_events[0].pos
        else:
            start_pos = filtered_events[0].pos
            end_pos = filtered_events[-1].pos
        transcript = Transcript(
            start=start_pos,
            end=end_pos + 1,    #end is exclusive
            strand=strand,
            events=tuple(filtered_events),
            score=score,
            exons=tuple(create_exons(filtered_events)),
        )
        transcripts.append(transcript)
    return transcripts


def filter_transcripts(transcripts: list[Transcript], min_score: float):
    """ This takes potential gene calls on both forward strand and reverse complement, filters them to remove
    overlapping calls, starting from highest scores first, and returns the best ones"""

    logger.info('Potential transcripts: %d', len(transcripts))
    valid_transcripts = [t for t in transcripts if t.score >= min_score]
    logger.info('Transcripts after filtering by min score: %d', len(valid_transcripts))
    if not valid_transcripts:
        return []

    valid_transcripts.sort(key=lambda x: x.start)
    non_overlapping = [valid_transcripts[0]]
    for t in valid_transcripts[1:]:
        if non_overlapping[-1].overlaps_with(t, ignore_strand=True):
            if t.score > non_overlapping[-1].score:
                non_overlapping[-1] = t
        else:
            non_overlapping.append(t)

    logger.info('Final transcripts after overlap removal: %d', len(non_overlapping))
    return non_overlapping


def build_transcripts(preds: Optional[np.ndarray], rc_preds: Optional[np.ndarray],
                     seq: str, rc_seq: Optional[str], contig_id: str, params: Params) -> list[Transcript]:
    """
    Build gene calls from a sequence using the GeneML model. Note that the coordinates in filtered_scored_gene_calls are
    relative to the sequence and the strand, so they are not absolute coordinates in the genome or even of the input
    sequence. See build_coords for converting to genomic absolute coordinates.
    """
    if preds is None and rc_preds is None:
        raise ValueError("Cannot build gene calls without any model predictions.")

    scored_gene_calls = None
    rc_scored_gene_calls = None
    transcripts = []
    seq_length = len(seq)
    if preds is not None:
        logger.info('Building gene calls on the forward strand')
        events = get_gene_ml_events(preds, params)
        scored_gene_calls = produce_gene_calls(preds, events, seq, contig_id + ' forward strand', params)
        transcripts.extend(create_transcripts(scored_gene_calls, seq_length, is_rc=False))

    if rc_preds is not None:
        logger.info('Building gene calls on the reverse strand')
        rc_events = get_gene_ml_events(rc_preds, params)
        rc_scored_gene_calls = produce_gene_calls(rc_preds, rc_events, rc_seq, contig_id + ' reverse strand', params)
        transcripts.extend(create_transcripts(rc_scored_gene_calls, seq_length, is_rc=True))

    logger.info('Selecting best gene calls for %s', contig_id)
    filtered_transcripts = filter_transcripts(
        transcripts, params.min_gene_score)

    return filtered_transcripts


def assign_transcripts_to_genes(transcripts_by_contig_id: dict[str, list[Transcript]]
                                ) -> dict[str, list[Gene]]:
    genes_by_contig = defaultdict(list)
    gene_count = 0
    for contig_id, transcripts in transcripts_by_contig_id.items():
        for transcript in transcripts:
            gene_count += 1
            if gene_count == 1e6:
                logger.warning('Reached 1 million predicted genes, '
                               'will produce gene IDs with more than 6 digits.')
            gene_id = f'GML{gene_count:06d}'
            gene = Gene(
                gene_id=gene_id,
                start=transcript.start,
                end=transcript.end,
                strand=transcript.strand,
                transcripts=(transcript,)
            )
            genes_by_contig[contig_id].append(gene)
    return genes_by_contig
