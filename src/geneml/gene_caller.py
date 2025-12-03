import logging
import time

import numpy as np
from numba import njit, objmode, typed, typeof

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

# Define the type for (score, gene_call) tuples used in select_gene_calls
ScoredGeneCallType = typeof((0.0, typed.List.empty_list(GeneEventNumbaType)))
# Define the type for (group_id, score, gene_call) tuples with group information
GroupedGeneCallType = typeof((0, 0.0, typed.List.empty_list(GeneEventNumbaType)))


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
            exon_size = exon_end - exon_start
            if exon_size < params.min_exon_size:
                continue
            if exon_size > params.max_exon_size:
                return num_ops      # Further exon ends make the exon even longer
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
                       starts_with_start_codon(new_cds) and
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
def score_gene_call(preds: np.ndarray, gene_call: list[GeneEvent]) -> float:
    """Compute a composite quality score for a complete gene call.

    Scores a gene structure by combining two metrics:
    A. Event consistency score: How well exonic/intronic regions match IS_EXON/IS_INTRON predictions
    B. Border score: Average confidence of the CDS and exon start and end predictions
    Score = (A + B) / 2

    Args:
        preds: Model predictions array with shape (num_features, sequence_length)
        gene_call: Complete gene structure as list of GeneEvents ordered by position

    Returns:
        Composite quality score for the gene call (float in range [0, 1])
    """
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
            num_vals += pos - (last_pos + 1)
        last_pos = pos
    event_consistency_score = 0.0
    if num_vals > 0:
        event_consistency_score = max(0.0, summed_scores / num_vals) # keep positive scores, else 0

    # Compute average score of the CDS and exon start and end predictions
    ends_score = 0.0
    for event in (gene_call[0], gene_call[-1]):
        ends_score += preds[EVENT_TYPE_MAP[event.type]][event.pos]
    ends_score = ends_score / 2

    # If there are internal events (splice sites), compute their average score
    if len(gene_call) > 2:
        splice_score = 0.0
        for i in range(1, len(gene_call) - 1):
            event = gene_call[i]
            splice_score += preds[EVENT_TYPE_MAP[event.type]][event.pos]
        splice_score = splice_score / (len(gene_call) - 2)
        # Balance splice score with ends score
        border_score = (splice_score + ends_score) / 2
    else:
        # Single-exon gene: only use ends_score
        border_score = ends_score

    score = (event_consistency_score + border_score) / 2
    return score


@njit
def compute_cds_length(gene_call: list[GeneEvent]) -> int:
    """Calculate the total coding sequence (CDS) length from a gene call.

    Sums the lengths of all exonic regions in a gene structure to determine
    the total CDS length.

    Args:
        gene_call: Gene structure as list of GeneEvents ordered by position

    Returns:
        Total CDS length (bp)
    """
    total_len = 0
    last_pos = -1

    for event in gene_call:
        pos = event.pos
        event_type = event.type

        if event_type in (CDS_START, EXON_START):
            last_pos = pos

        # gene_ml cds_end and exon_end predictions have off by one issue
        elif event_type in (CDS_END, EXON_END) and last_pos != -1:
            total_len += (pos + 1 - last_pos)
            last_pos = pos + 1
    return total_len


@njit
def select_gene_calls_per_group(group: list[tuple[float, list[GeneEvent]]],
                    max_transcripts: int) -> list[tuple[float, list[GeneEvent]]]:
    """Select best gene calls from a group of overlapping candidates.

    Filters a group of overlapping gene candidates to retain the most promising
    isoforms based on quality score and CDS length. Prioritizes keeping the longest
    gene among high-scoring candidates, plus additional high-scoring alternatives
    that are similar in length.

    Selection strategy:
    1. Identify best-scoring candidates (within a score threshold of top score)
    2. Keep the longest among those candidates
    3. Add top scorers that meet a minimum length requirement relative to the longest

    Args:
        group: List of (score, gene_call) tuples for overlapping gene candidates
        max_transcripts: Maximum number of alternative transcripts to retain

    Returns:
        List of (score, gene_call) tuples for selected gene calls
    """
    group.sort(key=lambda x: x[0], reverse=True) # sort by score
    best_score = group[0][0]
    best_candidates = [item for item in group if best_score - item[0] <= 0.2]

    # Keep longest candidate among best candidates
    lengths = np.array([compute_cds_length(call) for _, call in best_candidates])
    longest_i = np.argmax(lengths)
    longest = best_candidates[longest_i]
    keep = [longest]

    # Also keep top-scoring genes within 0.01 of best score
    # But only if they're at least 70% as long as the longest
    if max_transcripts > 1:
        longest_length = lengths[longest_i]
        min_length = longest_length * 0.7
        length_filtered = [item for item in group
                            if compute_cds_length(item[1]) >= min_length]

        # From length-filtered candidates, get top scorers within 0.01 of each other
        if length_filtered:
            top_score = length_filtered[0][0]
            top_scorers = [item for item in length_filtered
                            if top_score - item[0] <= 0.01
                            and item != longest]
            num_added_transcripts = min(len(top_scorers), max_transcripts - 1)
            keep.extend(top_scorers[:num_added_transcripts])

    return keep


@njit
def select_gene_calls(preds: np.ndarray, gene_calls: list[list[GeneEvent]],
                      min_score: float, max_transcripts: int,
                      ) -> list[tuple[int, float, list[GeneEvent]]]:
    """Select best gene calls from candidates, supporting alternative transcripts.

    Filters candidate gene calls based on minimum score.
    Groups overlapping gene calls and selects the most promising candidates from each
    group based on a combination of score and CDS length. Designed to retain both
    the longest isoform and high-scoring alternative transcripts.

    For each group of overlapping gene candidates:
        1. Sorts by score (descending)
        2. Identifies the call with longest CDS
        3. Keeps the longest plus top-scoring alternatives (up to max_transcripts total)

    Args:
        preds: Model predictions array with shape (num_features, sequence_length)
        gene_calls: List of candidate gene structures, each as a list of GeneEvents
        min_score: Minimum quality score threshold
        max_transcripts: Maximum number of alternative transcripts to retain per gene locus

    Returns:
        List of (group_id, score, gene_call) tuples for selected gene calls.
        group_id identifies which gene locus each call belongs to.
    """

    selected = typed.List.empty_list(GroupedGeneCallType)
    group = typed.List.empty_list(ScoredGeneCallType)
    group_id = 0

    for gene_call in gene_calls:
        score = score_gene_call(preds, gene_call)
        if score < min_score:
            continue

        if len(group) == 0:
            group.append((score, gene_call))
            continue

        last_call = group[-1][1]

        # Compare overlapping calls
        if gene_call[0].pos < last_call[-1].pos:
            group.append((score, gene_call))
        else:
            # Process completed group
            if len(group) > 1:
                best_calls = select_gene_calls_per_group(group, max_transcripts)
            else:
                best_calls = [group[0]]

            # Add group_id to each call - unpack the tuple to use its score
            for call_score, call in best_calls:
                selected.append((group_id, call_score, call))
            group_id += 1

            group.clear()
            group.append((score, gene_call))

    # Process final group
    if group:
        if len(group) > 1:
            best_calls = select_gene_calls_per_group(group, max_transcripts)
        else:
            best_calls = [group[0]]

        for call_score, call in best_calls:
            selected.append((group_id, call_score, call))

    return selected


@njit
def produce_gene_calls(preds: np.ndarray, events: list[GeneEvent], seq: str, contig_id: str,
                       params: Params) -> list[tuple[int, float, list[GeneEvent]]]:
    """ for a given set of events corresponding to a contig / candidate gene region, produce all possible gene calls"""
    function_start_time = python_time()
    start_time = python_time()
    num_ops = 0
    last_end_idx = -1
    skip_till_next_end_idx = False
    all_best_scores = typed.List.empty_list(GroupedGeneCallType)
    all_gene_calls = typed.List.empty_list(GeneCallNumbaType)
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

            # Remove all markers (single-event lists) from gene_calls
            if gene_calls:
                filtered_gene_calls = typed.List.empty_list(GeneCallNumbaType)
                for gene_call in gene_calls:
                    if len(gene_call) == 1:
                        # this is a marker for too many ops
                        elapsed = python_time() - recurse_start_time
                        if params.debug:
                            with objmode():
                                log = ' '.join([
                                    'too many ops for', contig_id, str(start_idx), str(end_idx),
                                    prettify_gene_event(start_event), prettify_gene_event(events[end_idx]),
                                    prettify_gene_event(gene_call[0]),
                                    '; elapsed time:', str(np.round(elapsed, 2))])
                                logger.debug(log)
                        # skip this marker, don't add to filtered list
                        skip_till_next_end_idx = True
                    else:
                        filtered_gene_calls.append(gene_call)
                gene_calls = filtered_gene_calls

            # Sort by start position, then end position
            gene_calls.sort(key=lambda x: (x[0].pos, x[-1].pos))
            all_gene_calls.extend(gene_calls)

            if num_ops > params.single_recurse_max_num_ops:
                with objmode():
                    elapsed = python_time() - start_time
                    log = ' '.join(['num_ops exceeded (' + str(num_ops) + ', ' + str(round(elapsed, 1)) + 's) for',
                                    contig_id, str(start_idx), str(end_idx),
                                    prettify_gene_event(start_event), prettify_gene_event(events[end_idx])])
                    logger.info(log)

                # short circuit this gene region
                skip_till_next_end_idx = True

    elapsed = python_time() - function_start_time
    if elapsed > 600:
        with objmode():
            log = 'slow {contig_id} at {elapsed:.2f}s'.format(contig_id=contig_id, elapsed=elapsed)
            logger.warning(log)

    if all_gene_calls:
        if params.dynamic_scoring:
            min_score = 0.01
        else:
            min_score = params.min_gene_score
        all_best_scores = select_gene_calls(preds, all_gene_calls, min_score,
                                            params.max_transcripts)
    return all_best_scores
