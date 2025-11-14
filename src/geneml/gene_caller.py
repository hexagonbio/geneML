import logging
import time

import numpy as np
from numba import njit, objmode, typed

from geneml.model_loader import (
    MODEL_CDS_END,
    MODEL_CDS_START,
    MODEL_EXON_END,
    MODEL_EXON_START,
    MODEL_IS_EXON,
    MODEL_IS_INTRON,
)
from geneml.params import Params
from geneml.types import (
    CDS_END,
    CDS_START,
    EXON_END,
    EXON_START,
    GeneCallNumbaType,
    GeneEvent,
    GeneEventNumbaType,
)

logger = logging.getLogger("geneml")

EVENT_TYPE_MAP = (MODEL_CDS_START, MODEL_CDS_END, MODEL_EXON_START, MODEL_EXON_END)


@njit
def python_time() -> float:
    with objmode(out='float64'):
        out = time.time()
    return out


@njit
def prettify_gene_event(event: GeneEvent) -> str:
    with objmode(out='unicode_type'):
        out = '{pos}:{type}:{score:.1f}'.format(pos=event.pos, type=event.type, score=event.score)
    return out


@njit
def get_gene_ml_events(preds: np.ndarray, params: Params):
    cds_starts = np.where(preds[MODEL_CDS_START] >= params.cds_start_min_score)[0]
    cds_ends = np.where(preds[MODEL_CDS_END] >= params.cds_end_min_score)[0]
    exon_starts = np.where(preds[MODEL_EXON_START] >= params.exon_start_min_score)[0]
    exon_ends = np.where(preds[MODEL_EXON_END] >= params.exon_end_min_score)[0]

    events = typed.List.empty_list(GeneEventNumbaType)
    events.extend([GeneEvent(i, CDS_START, preds[MODEL_CDS_START, i]) for i in cds_starts])
    events.extend([GeneEvent(i, CDS_END, preds[MODEL_CDS_END, i]) for i in cds_ends])
    events.extend([GeneEvent(i, EXON_START, preds[MODEL_EXON_START, i]) for i in exon_starts])
    events.extend([GeneEvent(i, EXON_END, preds[MODEL_EXON_END, i]) for i in exon_ends])
    events.sort()

    return events


@njit
def filter_events(one_gene_events: list[GeneEvent], percentile_cutoff: int,
                  min_exon_events: int, max_exon_events: int) -> list[GeneEvent]:
    """Filter gene events using percentile-based score thresholds.

    Reduces the event set for a single gene region to limit the number of recursions.
    Removes:
        - Any CDS_START events that are not the first event
        - EXON_START and EXON_END events below a score percentile threshold (capped at 0.1 minimum)
    Retains all CDS_END events.
    Only performs filtering on EXON_START and EXON_END events if there are more events
    than the specified min_exon_events.
    A hard maximum of max_exon_events is also enforced.

    Args:
        one_gene_events: List of GeneEvent objects for a single gene region
        percentile_cutoff: Percentile threshold (0-100) for filtering event scores
        min_exon_events: Minimum number of exon events to retain per type
        max_exon_events: Maximum number of exon events to retain per type

    Returns:
        Filtered list of GeneEvent objects for the gene region
    """
    assert one_gene_events[0].type == CDS_START, 'first event must be CDS_START'

    def filter_exon_events(events: list[GeneEvent], percentile: int,
                           min_events: int, max_events: int) -> list[GeneEvent]:
        percentile_cutoff = np.percentile([e.score for e in events], percentile)
        threshold = min(percentile_cutoff, 0.1)
        filtered = [e for e in events if e.score >= threshold]
        if len(filtered) < min_events:
            return events[:min_events]
        if len(filtered) > max_events:
            return events[:max_events]
        return filtered

    cds_start = [one_gene_events[0]]
    other_events = []

    for exon_event in [EXON_START, EXON_END, CDS_END]:
        events = sorted([e for e in one_gene_events[1:] if e.type == exon_event],
                             key=lambda x: x.score, reverse=True)
        if events:
            if exon_event in [EXON_START, EXON_END]:
                filtered = filter_exon_events(events, percentile_cutoff,
                                            min_exon_events, max_exon_events)
                other_events.extend(filtered)
            else:
                # Keep all CDS_END events without filtering
                other_events.extend(events)

    other_events.sort(key=lambda x: x.pos)

    return cds_start + other_events


@njit
def get_end_idx(start_idx: int, events: list[GeneEvent], preds: np.ndarray) -> int:
    event = events[start_idx]
    start_pos = event.pos
    pos = start_pos
    num_good_bases = 0
    last_good_base = None
    while pos < len(preds[0]):
        if preds[MODEL_IS_EXON, pos] > 0.2 or preds[MODEL_IS_INTRON, pos] > 0.2:
            num_good_bases += 1
            last_good_base = pos
        else:
            if pos - start_pos > 300 and (num_good_bases / (pos - start_pos) < 0.7 or pos - last_good_base > 200):
                break
        pos += 1

    for i in range(start_idx, len(events)):
        if events[i].pos >= pos:
            return i
    return len(events) - 1


@njit
def starts_with_start_codon(seq):
    # note: assumes seq is all uppercase
    if len(seq) < 3:
        return False
    codon = seq[0:3]
    return codon in ('ATG', 'TTG', 'CTG')


@njit
def count_stop_codons(seq):
    # note: assumes seq is all uppercase
    count = 0
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in ('TAA', 'TAG', 'TGA'):
            count += 1
    return count


@njit
def ends_with_stop_codon(seq):
    # note: assumes seq is all uppercase
    if len(seq) < 3:
        return False
    codon = seq[-3:]
    return codon in ('TAA', 'TAG', 'TGA')


@njit
def check_exon_validity(preds: np.ndarray, start: int, end: int,
                        min_avg_exon_score: float, max_intron_score: float) -> bool:
    """Check if an exon region is consistent with exonic predictions.

    Validates that a proposed exon is supported by the underlying model predictions.
    Checks that:
    1. No IS_INTRON score in the region exceeds the maximum threshold
    2. Average IS_EXON score across the region meets the minimum threshold

    Args:
        preds: Model predictions array with shape (num_features, sequence_length)
        start: Start position of the exon
        end: End position of the exon
        min_avg_exon_score: Minimum average IS_EXON score required across the region
        max_intron_score: Maximum IS_INTRON score allowed at any position in the region

    Returns:
        Boolean value indicating whether the exon passes consistency checks
    """
    length = end - start
    if not length:
        return False

    intron_scores = preds[MODEL_IS_INTRON, start:end]
    exon_scores = preds[MODEL_IS_EXON, start:end]

    if np.any(intron_scores > max_intron_score):
        return False

    if np.mean(exon_scores) < min_avg_exon_score:
        return False

    return True


@njit
def recurse(results: list[list[GeneEvent]], events: list[GeneEvent], i: int, gene: list, seq: str,
            preds: np.ndarray, params: Params, current_cds: str = "") -> int:
    """Recursively build valid gene structures from filtered gene events.

    Explores all valid combinations of gene events (CDS_START, CDS_END, EXON_START, EXON_END)
    to construct complete gene structures. Uses depth-first recursion with early pruning based
    on biological constraints and quality checks. Incrementally builds and validates the coding
    sequence (CDS) to detect invalid paths early.

    Validation checks applied during recursion:
    - Intron and exon size constraints
    - Exon region consistency with IS_EXON/IS_INTRON predictions
    - Start codon presence at beginning of CDS
    - Premature stop codons in partial CDS
    - Valid gene structure: exactly one stop codon at the end, divisible by 3

    The recursion is bounded by operation limits to prevent excessive computation in complex
    regions. When limits are exceeded, a marker is added to results to signal truncation.

    Args:
        results: Output list to accumulate valid gene structures (modified in-place)
        events: Filtered list of gene events to explore, starting with CDS_START
        i: Current index in the events list being processed
        gene: Current partial gene structure being built (modified during recursion)
        seq: DNA sequence for the genomic region
        preds: Model predictions array with shape (num_features, sequence_length)
        params: Configuration parameters
        current_cds: Incrementally built coding sequence for early validation

    Returns:
        Total number of recursive operations performed (for budget tracking)
    """
    num_ops = 0

    # handle beginning of gene
    if i == 0:
        assert events[0].type == CDS_START, 'events must start with a cds_start event'
        gene.append(events[i])
        i += 1

    # Nothing to do if index out of range
    if i >= len(events):
        return 0

    # Loop over remaining events
    for j in range(i, len(events)):
        event = events[j]
        last_type = gene[-1].type

        if last_type in {CDS_START, EXON_START} and event.type not in {EXON_END, CDS_END}:
            continue  # can only add an end after a start
        if last_type == EXON_END and event.type != EXON_START:
            continue  # can only add a start after an end

        new_cds = current_cds

        # If we're starting an exon, check the size of the previous intron
        if event.type == EXON_START:
            intron_start = gene[-1].pos + 1
            intron_end = event.pos
            intron_size = intron_end - intron_start
            if intron_size < params.min_intron_size:
                continue
            if intron_size > params.max_intron_size:
                return num_ops      # Further exon starts make the intron even longer

        # If we're closing an exon, check consistency and validity
        elif event.type in {EXON_END, CDS_END}:
            exon_start = gene[-1].pos
            exon_end = event.pos + 1
            if not check_exon_validity(preds, exon_start, exon_end,
                                       min_avg_exon_score=0.05, max_intron_score=0.9):
                return num_ops      # Further exon ends are also expected to be inconsistent

            # Build CDS incrementally - extract current exon sequence
            exon_seq = seq[exon_start:exon_end].upper()
            new_cds = current_cds + exon_seq

            # For EXON_END, check for premature stop codons to prune invalid paths early
            if event.type == EXON_END:
                num_stop_codons = count_stop_codons(new_cds)
                if len(new_cds) > 2 and not starts_with_start_codon(new_cds):
                    return num_ops  # Further exon ends (with same start) will also lack start codon
                if num_stop_codons > 0:
                    return num_ops  # Further exon ends (with same start) will also contain stop codons

        # If previous checks passed, add the event to the gene
        gene.append(event)

        if event.type == CDS_END:
            # Final validity check using cached CDS
            num_stop_codons = count_stop_codons(new_cds)
            is_valid = (num_stop_codons == 1 and
                       len(new_cds) % 3 == 0 and
                       ends_with_stop_codon(new_cds))
            if is_valid:
                results.append(gene.copy())
                if len(results) >= params.gene_candidates:
                    gene.pop()
                    break
            gene.pop()  # Remove the cds end event and look for other possibilities
            continue

        if num_ops <= params.single_recurse_max_num_ops:
            num_ops += 1
            num_ops += recurse(results, events, j + 1, gene, seq, preds, params, new_cds)
            gene.pop()
        else:
            marker = typed.List.empty_list(GeneEventNumbaType)
            marker.append(event)
            results.append(marker)  # marker for too many ops
            gene.pop()
            break

    return num_ops


@njit
def score_gene_call(preds: np.ndarray, gene_call: list[GeneEvent], seq: str):
    # look at the is_exon/is_intron scores based on the intron boundaries defined by the gene call
    last_pos = None
    summed_scores = 0
    num_vals = 0
    for pos, event_type, score in gene_call:
        if last_pos is not None:
            is_exon = event_type in (CDS_END, EXON_END)
            key = MODEL_IS_EXON if is_exon else MODEL_IS_INTRON
            other_key = MODEL_IS_INTRON if is_exon else MODEL_IS_EXON
            summed_scores += np.sum(preds[key, last_pos+1:pos]-preds[other_key, last_pos+1:pos])
            summed_scores += score  # include the score of the event itself
            num_vals += pos - last_pos
        last_pos = pos
    event_consistency_score = 0.0
    if num_vals:
        event_consistency_score = (summed_scores / num_vals + 1) / 2 # scale from range -1,1 to range 0,1

    # cds start and end, exon start and end--the more/higher the more confident we are
    cds_ends_score = 0
    for e in (gene_call[0], gene_call[-1]):
        cds_ends_score += preds[EVENT_TYPE_MAP[e.type]][e.pos] / 2 # scale from range 0,2 to range 0,1

    gene_length_score = (gene_call[-1].pos - gene_call[0].pos) / 10000  # slight preference for longer genes

    score = (
        event_consistency_score +
        cds_ends_score +
        gene_length_score
    )
    return score


@njit
def produce_gene_calls(preds: np.ndarray, events: list[GeneEvent], seq: str, contig_id: str, params: Params) -> list[tuple[float, list[GeneEvent]]]:
    """ for a given set of events corresponding to a contig / candidate gene region, produce all possible gene calls"""
    function_start_time = python_time()
    start_time = python_time()
    num_ops = 0
    last_end_idx = -1
    skip_till_next_end_idx = False
    all_best_scores = []
    for start_idx, start_event in enumerate(events):
        if start_event.type == CDS_START:
            end_idx = get_end_idx(start_idx, events, preds)
            if end_idx - start_idx < 2:
                continue

            # check if range start_idx:end_idx contains a cds_end event
            cds_end_found = False
            for e in events[start_idx:end_idx+1]:
                if e.type == CDS_END:
                    cds_end_found = True
                    if end_idx != last_end_idx:
                        last_end_idx = end_idx
                        start_time = python_time()
                        num_ops = 0
                        skip_till_next_end_idx = False
                    elif num_ops > params.recurse_region_max_num_ops:
                        with objmode():
                            log = ' '.join([
                                'recurse_region_max_num_ops reached', str(num_ops), str(python_time() - start_time),
                                contig_id, str(start_idx), str(end_idx),
                                prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx])])
                            logger.info(log)
                        skip_till_next_end_idx = True
                    break
            if not cds_end_found or skip_till_next_end_idx:
                # skip this recurse region
                continue

            one_gene_events = events[start_idx:end_idx+1]
            filtered_events = filter_events(one_gene_events, percentile_cutoff=60,
                                            min_exon_events=15, max_exon_events=60)
            gene_calls = typed.List.empty_list(GeneCallNumbaType)
            recurse_start_time = python_time()
            num_ops += recurse(gene_calls, filtered_events, 0,
                               typed.List.empty_list(GeneEventNumbaType), seq, preds, params, "")

            if params.debug:
                with objmode:
                    log = ' '.join([str(start_idx), str(end_idx), prettify_gene_event(start_event), prettify_gene_event(events[end_idx]),
                      ';', str(len(gene_calls)), 'gene calls',
                      ])
                    logger.debug(log)

            if gene_calls and len(gene_calls[-1]) == 1:
                # this is a marker for too many ops
                elapsed = python_time() - recurse_start_time
                with objmode():
                    log = ' '.join([
                        'too many ops for', contig_id, str(start_idx), str(end_idx),
                        prettify_gene_event(start_event), prettify_gene_event(events[end_idx]),
                        prettify_gene_event(gene_calls[-1][0]),
                        '; elapsed time:', str(np.round(elapsed, 2))])
                    logger.debug(log)

                # remove the last gene call marker
                gene_calls = gene_calls[:-1]

                # short circuit this gene region
                skip_till_next_end_idx = True

            if gene_calls:

                scores = []
                for gene_call in gene_calls:
                    scores.append((score_gene_call(preds, gene_call, seq), gene_call))
                scores.sort(reverse=True)

                # choose just the best... revisit?
                all_best_scores.append(scores[0])

            if num_ops > params.single_recurse_max_num_ops:
                with objmode():
                    elapsed = python_time() - start_time
                    log = ' '.join(['num_ops exceeded (' + str(num_ops) + ', ' + str(round(elapsed, 1)) + 's) for',
                                    contig_id, str(start_idx), str(end_idx),
                                    prettify_gene_event(start_event), prettify_gene_event(events[end_idx])])
                    logger.info(log)

                # short circuit this gene region
                skip_till_next_end_idx = True
    all_best_scores.sort(reverse=True)

    elapsed = python_time() - function_start_time
    if elapsed > 600:
        with objmode():
            log = 'slow {contig_id} at {elapsed:.2f}s'.format(contig_id=contig_id, elapsed=elapsed)
            logger.warning(log)

    return all_best_scores
