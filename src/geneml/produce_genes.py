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


def create_exons(gene_events: list[GeneEvent], contig_length: int, is_rc: bool) -> list[Exon]:
    if not gene_events:
        return []
    assert len(gene_events) % 2 == 0, f'There should be an even number of gene events: {gene_events}'
    exons = []
    for event, next_event in zip(gene_events[::2], gene_events[1::2]):
        if not (event.type in (CDS_START, EXON_START) and next_event.type in (CDS_END, EXON_END)):
            raise ValueError(f'Invalid gene events pair (expected start event and end event): {event}, {next_event}')
        if is_rc:
            start_pos = contig_length - next_event.pos - 1
            end_pos = contig_length - event.pos - 1
        else:
            start_pos = event.pos
            end_pos = next_event.pos
        exon = Exon(
            start=start_pos,
            end=end_pos + 1,  # end is exclusive
            events=(event, next_event)
        )
        exons.append(exon)
    return exons


def create_transcripts(scored_gene_calls: list[tuple[int, float, list[GeneEvent]]],
                       contig_length: int, is_rc: bool) -> list[Transcript]:
    transcripts = []
    strand = -1 if is_rc else 1
    for group_id, score, gene_events in scored_gene_calls:
        if is_rc:
            start_pos = contig_length - gene_events[-1].pos - 1
            end_pos = contig_length - gene_events[0].pos - 1
        else:
            start_pos = gene_events[0].pos
            end_pos = gene_events[-1].pos
        transcript = Transcript(
            start=start_pos,
            end=end_pos + 1,    #end is exclusive
            strand=strand,
            events=tuple(gene_events),
            score=score,
            exons=tuple(create_exons(gene_events, contig_length, is_rc)),
            group_id=group_id,
        )
        transcripts.append(transcript)
    return transcripts


def filter_transcripts(transcripts: list[Transcript], min_score: float, contig_id: str
                       ) -> list[Transcript]:
    """ This takes potential gene calls on both forward strand and reverse complement, filters them to remove
    overlapping calls, starting from highest scores first, and returns the best ones"""

    valid_transcripts = [t for t in transcripts if t.score >= min_score]
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

    logger.info('%s: Potential transcripts: %d', contig_id, len(transcripts))
    logger.info('%s: Transcripts after filtering by min score: %d', contig_id, len(valid_transcripts))
    logger.info('%s: Final transcripts after overlap removal: %d', contig_id, len(non_overlapping))

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
        logger.info('%s 3/5: Building gene calls on forward strand', contig_id)
        events = get_gene_ml_events(preds, params)
        scored_gene_calls = produce_gene_calls(preds, events, seq, contig_id + ' forward strand', params)
        if scored_gene_calls:
            transcripts.extend(create_transcripts(scored_gene_calls, seq_length, is_rc=False))

    if rc_preds is not None:
        logger.info('%s 4/5: Building gene calls on reverse strand', contig_id)
        rc_events = get_gene_ml_events(rc_preds, params)
        rc_scored_gene_calls = produce_gene_calls(rc_preds, rc_events, rc_seq, contig_id + ' reverse strand', params)
        if rc_scored_gene_calls:
            transcripts.extend(create_transcripts(rc_scored_gene_calls, seq_length, is_rc=True))

    logger.info('%s 5/5: Selecting best gene calls', contig_id)
    filtered_transcripts = filter_transcripts(
        transcripts, params.min_gene_score, contig_id)

    return filtered_transcripts


def assign_transcripts_to_genes(transcripts_by_contig_id: dict[str, list[Transcript]]
                                ) -> dict[str, list[Gene]]:
    """Assign transcripts to genes and generate unique gene identifiers.

    Groups transcripts into gene loci based on the group_id assigned during gene calling.
    Transcripts sharing the same contig, strand, and group_id are considered alternative
    isoforms of the same gene. Each gene is assigned a unique sequential identifier in
    the format 'GML######'.

    Args:
        transcripts_by_contig_id: Dictionary mapping contig IDs to lists of Transcript objects

    Returns:
        Dictionary mapping contig IDs to lists of Gene objects, where each Gene contains
        one or more alternative transcript isoforms
    """
    genes_by_contig = defaultdict(list)
    gene_count = 0
    transcript_count = 0

    for contig_id, transcripts in transcripts_by_contig_id.items():
        if not transcripts:
            continue

        transcript_count += len(transcripts)

        # Group by (strand, group_id) to create genes
        gene_groups = defaultdict(list)
        for t in transcripts:
            key = (t.strand, t.group_id)
            gene_groups[key].append(t)

        # Sort groups by start position
        sorted_groups = sorted(gene_groups.items(), key=lambda x: x[1][0].start)

        # Create Gene objects
        for _, group in sorted_groups:
            gene_count += 1
            if gene_count == 1_000_000:
                logger.warning('Reached 1 million predicted genes, '
                               'will produce gene IDs with more than 6 digits.')
            gene = Gene(
                gene_id=f'GML{gene_count:06d}',
                start=group[0].start,
                end=max(t.end for t in group),
                strand=group[0].strand,
                transcripts=tuple(group),
            )
            genes_by_contig[contig_id].append(gene)

    logger.info('Total predicted transcripts: %d', transcript_count)
    logger.info('Total predicted genes: %d (%.2f transcripts per gene)',
                gene_count, transcript_count / gene_count if gene_count > 0 else 0)
    return genes_by_contig
