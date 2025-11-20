import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from geneml.gene_caller import get_gene_ml_events, produce_gene_calls, score_gene_call
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


def create_exons(gene_events: list[GeneEvent], preds: np.ndarray,
                 contig_length: int, is_rc: bool) -> list[Exon]:
    if not gene_events:
        return []
    assert len(gene_events) % 2 == 0, f'There should be an even number of gene events: {gene_events}'
    exons = []
    frame = 0  # Tracks the reading frame position at the end of the previous exon
    for event, next_event in zip(gene_events[::2], gene_events[1::2]):
        if not (event.type in (CDS_START, EXON_START) and next_event.type in (CDS_END, EXON_END)):
            raise ValueError(f'Invalid gene events pair (expected start event and end event): {event}, {next_event}')
        if is_rc:
            start_pos = contig_length - next_event.pos - 1
            end_pos = contig_length - event.pos - 1
        else:
            start_pos = event.pos
            end_pos = next_event.pos

        # Phase indicates how many bases from the previous exon are needed to complete a codon
        phase = (3 - frame) % 3

        exon = Exon(
            start=start_pos,
            end=end_pos + 1,  # end is exclusive
            events=(event, next_event),
            score=score_gene_call(preds, [event, next_event]),
            phase=phase,
        )
        exons.append(exon)
        exon_length = exon.end - exon.start
        frame = (frame + exon_length) % 3
    return exons


def create_transcripts(scored_gene_calls: list[tuple[int, float, list[GeneEvent]]],
                       preds: np.ndarray, contig_length: int, is_rc: bool) -> list[Transcript]:
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
            exons=tuple(create_exons(gene_events, preds, contig_length, is_rc)),
            group_id=group_id,
        )
        transcripts.append(transcript)
    return transcripts


def filter_overlapping_transcripts(transcripts: list[Transcript]) -> list[Transcript]:
    """ This filters overlapping transcripts on opposite strands, keeping the higher scoring one """

    if not transcripts:
        return []

    non_overlapping = [transcripts[0]]
    for t in transcripts[1:]:
        if (t.strand != non_overlapping[-1].strand and
            non_overlapping[-1].overlaps_with(t, ignore_strand=True) and
            t.score > non_overlapping[-1].score):
            non_overlapping[-1] = t
            continue
        non_overlapping.append(t)

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
            transcripts.extend(create_transcripts(scored_gene_calls, preds, seq_length, is_rc=False))

    if rc_preds is not None:
        logger.info('%s 4/5: Building gene calls on reverse strand', contig_id)
        rc_events = get_gene_ml_events(rc_preds, params)
        rc_scored_gene_calls = produce_gene_calls(rc_preds, rc_events, rc_seq, contig_id + ' reverse strand', params)
        if rc_scored_gene_calls:
            transcripts.extend(create_transcripts(rc_scored_gene_calls, rc_preds, seq_length, is_rc=True))

    logger.info('%s 5/5: Selecting best gene calls', contig_id)

    transcripts.sort(key=lambda x: (x.start, x.end))

    if not params.allow_opposite_strand_overlaps:
        transcripts = filter_overlapping_transcripts(transcripts)

    return transcripts


def collect_all_scores(transcripts_by_contig_id: dict[str, list[Transcript]]
                           ) -> list[float]:
    """Collect all transcript scores from all contigs into a single list."""
    all_scores = []
    for transcripts in transcripts_by_contig_id.values():
        all_scores.extend([t.score for t in transcripts])
    return all_scores


def build_histogram(scores: np.ndarray, num_bins: int, smoothing_window: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Build a histogram of transcript scores.

    Args:
        scores: 1D numpy array of transcript scores
        num_bins: Number of bins to use for the histogram
        smoothing_window: Window size for smoothing the histogram

    Returns:
        hist: Counts for each bin
        bin_centers: Centers of the bins
    """
    hist, bin_edges = np.histogram(scores, bins=num_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth the histogram with a rolling average to reduce noise
    window_size = smoothing_window
    smoothed_hist = np.convolve(hist, np.ones(window_size)/window_size, mode='same')

    return smoothed_hist, bin_centers


def get_dynamic_threshold(transcripts_by_contig_id: dict[str, list[Transcript]],
                          fallback_threshold: float, min_threshold: float = 0.2,
                          max_threshold: float = 0.8) -> float:
    """Determine a dynamic minimum gene score threshold based on the distribution of transcript
    scores across all contigs.

    Finds the deepest valley between two peaks in a bimodal score distribution. The distribution
    is expected to have two major peaks (low-confidence and high-confidence predictions), and this
    function identifies the threshold at the deepest valley between them, constrained to the
    [min_threshold, max_threshold] range.

    Args:
        transcripts_by_contig_id: Dictionary mapping contig IDs to lists of Transcript objects
        fallback_threshold: Threshold to use if dynamic calculation is not possible
        min_threshold: Minimum allowed threshold value (default: 0.2)
        max_threshold: Maximum allowed threshold value (default: 0.8)

    Returns:
        Float threshold value representing the score at the deepest valley within
        [min_threshold, max_threshold]. Falls back to fallback_threshold if insufficient
        data or no valid valley is found in the allowed range.
    """
    # Collect all transcript scores
    all_scores = collect_all_scores(transcripts_by_contig_id)
    num_scores = len(all_scores)
    if num_scores < 100:
        logger.warning(
            'Insufficient transcripts (%d) for dynamic scoring, using fallback threshold of %.1f',
            num_scores, fallback_threshold
        )
        return fallback_threshold

    # Set number of bins using square-root rule, capped between 10 and 150
    num_bins = min(150, max(10, int(np.sqrt(num_scores))))

    # Set smoothing window adaptively: use ~20% of bins, capped between 5 and 30
    smoothing_window = min(30, max(5, int(num_bins * 0.2)))

    # Create histogram to smooth the distribution
    scores_array = np.array(all_scores)
    smoothed_hist, bin_centers = build_histogram(scores_array, num_bins=num_bins,
                                                 smoothing_window=smoothing_window)

    # Find local minima (valleys) and maxima (peaks)
    diff = np.diff(smoothed_hist)

    # Find indices where derivative crosses zero
    minima_indices = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1
    maxima_indices = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1

    if len(maxima_indices) < 2 or len(minima_indices) == 0:
        logger.warning(
            'No clear multimodal distribution found (peaks: %d, valleys: %d), '
            'using fallback threshold of %.1f',
            len(maxima_indices), len(minima_indices), fallback_threshold
        )
        return fallback_threshold

    # For each minimum, check if it's between two peaks AND within the allowed threshold range
    valid_minima = []
    for minimum_idx in minima_indices:
        threshold_value = bin_centers[minimum_idx]

        # Skip minima outside the allowed range
        if threshold_value < min_threshold or threshold_value > max_threshold:
            continue

        peaks_left = maxima_indices[maxima_indices < minimum_idx]
        peaks_right = maxima_indices[maxima_indices > minimum_idx]

        if len(peaks_left) > 0 and len(peaks_right) > 0:
            valid_minima.append(minimum_idx)

    if len(valid_minima) == 0:
        logger.warning(
            'No local minimum found between peaks in range [%.2f, %.2f], '
            'using fallback threshold of %.1f',
            min_threshold, max_threshold, fallback_threshold
        )
        return fallback_threshold

    # Find the lowest valid minimum
    minimum_depths = smoothed_hist[valid_minima]
    lowest_minimum_idx = valid_minima[np.argmin(minimum_depths)]

    threshold = round(bin_centers[lowest_minimum_idx], 2)

    logger.info(
        'Dynamic threshold calculated: %.2f (based on %d transcripts)',
        threshold, len(all_scores)
    )

    return float(threshold)


def filter_by_dynamic_threshold(transcripts_by_contig_id: dict[str, list[Transcript]],
                                fallback_threshold: float) -> dict[str, list[Transcript]]:
    """Filter transcripts by a dynamically determined score threshold.

    Args:
        transcripts_by_contig_id: Dictionary mapping contig IDs to lists of Transcript objects
        fallback_threshold: Threshold to use if dynamic calculation is not possible

    Returns:
        Dictionary mapping contig IDs to lists of Transcript objects that pass the dynamic threshold
    """
    threshold = get_dynamic_threshold(transcripts_by_contig_id, fallback_threshold)

    filtered_transcripts_by_contig_id = {}
    for contig_id, transcripts in transcripts_by_contig_id.items():
        filtered_transcripts = [t for t in transcripts if t.score >= threshold]
        filtered_transcripts_by_contig_id[contig_id] = filtered_transcripts

    return filtered_transcripts_by_contig_id


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
