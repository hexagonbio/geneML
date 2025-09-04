import numpy as np
import time
from Bio.Seq import reverse_complement
from collections import namedtuple
from numba import njit, objmode, typed, typeof, types

from geneml.model_loader import (
    MODEL_CDS_END,
    MODEL_CDS_START,
    MODEL_EXON_END,
    MODEL_EXON_START,
    MODEL_IS_EXON,
    MODEL_IS_INTRON,
    ResidualModelBase,
)
from geneml.params import Params
from geneml.utils import chunked_seq_predict

# using dataclass would be nice, but numba doesn't support it
GeneEvent = namedtuple('GeneEvent', ['pos', 'type', 'score'])

CDS_START = 0
CDS_END = 1
EXON_START = 2
EXON_END = 3

EVENT_TYPE_MAP = (MODEL_CDS_START, MODEL_CDS_END, MODEL_EXON_START, MODEL_EXON_END)

GeneEventNumbaType = typeof(GeneEvent(1, CDS_START, np.float32(0.5)))
GeneCallNumbaType = typeof(typed.List.empty_list(GeneEventNumbaType))


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
def get_gene_ml_events(preds: np.ndarray, params: namedtuple):
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
def get_end_idx(start_idx: int, events: list[GeneEvent], preds: np.ndarray) -> int:
    event = events[start_idx]
    start_pos = event.pos
    pos = start_pos
    num_good_bases = 0
    last_good_base = None
    while pos < len(preds[0]):
        if preds[MODEL_IS_EXON, pos] > 0.1 or preds[MODEL_IS_INTRON, pos] > 0.1:
            num_good_bases += 1
            last_good_base = pos
        else:
            if pos - start_pos > 300 and (num_good_bases / (pos - start_pos) < 0.7 or pos - last_good_base > 200):
                break
        pos += 1

    # print(start_pos, pos)
    for i in range(start_idx, len(events)):
        if events[i].pos >= pos:
            return i
    return len(events) - 1


@njit
def count_stop_codons(seq):
    count = 0
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in ('TAA', 'TAG', 'TGA'):
            count += 1
    return count


@njit
def ends_with_stop_codon(seq):
    if len(seq) < 3:
        return False
    codon = seq[-3:]
    return codon in ('TAA', 'TAG', 'TGA')


@njit
def check_sequence_validity(gene_call: list[GeneEvent], seq: str) -> bool | None:
    assert gene_call and gene_call[-1].type in {EXON_END, CDS_END}, 'validity check only works with gene_calls that end with exon_end or cds_end'

    cds_seq = build_cds_seq(seq, gene_call)
    num_stop_codons = count_stop_codons(cds_seq)

    # partial sequence so just check for premature stop codon
    if gene_call[-1].type == EXON_END:
        return num_stop_codons == 0

    # full length sequence so also check for stop codon and multiple of three
    elif gene_call[-1].type == CDS_END:
        return num_stop_codons == 1 and len(cds_seq) % 3 == 0 and ends_with_stop_codon(cds_seq)

    else:
        assert False, "shouldn't get here"


@njit
def rerank_indices_based_on_most_likely_next_events(gene: list, events: list[GeneEvent], start_idx: int, params: Params) -> list[int]:
    """ Rerank the indices of the events list based on the most likely next events, to increase chance of identifying
    the best gene call based on the prediction scores within the earlier parts of the recursion
    """
    enumerated_events = list(enumerate(events))
    end_idx = len(events)
    if gene[-1].type in (CDS_START, EXON_START):
        candidate_exon_end_indices = [i for i, e in enumerated_events[start_idx:] if e.type == EXON_END and e.score > 0.2]
        candidate_exon_end_indices = candidate_exon_end_indices[:2]
        priority_indices = candidate_exon_end_indices

        # TODO: want to make sure some cds_end events are considered early, but for some reason this makes a lot of genes disappear
        # if len(gene) <= 3:
        #     candidate_cds_end_enumerated_events = [(i, e) for i, e in enumerated_events[start_idx:] if e.type == CDS_END]
        #     candidate_cds_end_enumerated_events.sort(key=lambda x: x[1].score, reverse=True)
        #     priority_indices = [i for i, e in candidate_cds_end_enumerated_events[:5]] + priority_indices

    elif gene[-1].type == EXON_END:
        end_idx = start_idx
        while events[end_idx].pos < gene[-1].pos + params.max_intron_size and end_idx < len(events) - 1:
            end_idx += 1
        priority_indices = [i for i, e in enumerated_events[start_idx:end_idx] if e.type == EXON_START and e.score > 0.2]
    else:
        assert False, 'invalid gene event type'

    # for completeness (even though the next event type might be constrained), add the remaining indices
    # in the order they appear in the events list
    indices = priority_indices + [i for i in range(start_idx, end_idx) if i not in priority_indices]
    return indices


@njit
def recurse(results: list[list[GeneEvent]], events: list[GeneEvent], i: int, gene: list, seq: str,
            params: namedtuple) -> int:
    """ Recursively attempt to build genes from the events list
    """
    num_ops = 1

    # handle beginning of gene
    if i == 0:
        assert events[0].type == CDS_START, 'events must start with a cds_start event'
        gene.append(events[i])
        i += 1

    indices = rerank_indices_based_on_most_likely_next_events(gene, events, i, params)
    for i in indices:
        if len(results) >= params.gene_candidates or results and len(results[-1]) == 1:
            # limit the number of results to speed up
            break

        event = events[i]
        new_gene = copy_and_append_gene_event_numba(gene, event)

        # handle cds end
        if gene and event.type == CDS_END and check_sequence_validity(new_gene, seq):
            results.append(new_gene)

        if i + 1 >= len(events):
            break  # no more additional events to consider

        # handle intron start (exon end)
        if gene[-1].type in {CDS_START, EXON_START} and event.type == EXON_END and check_sequence_validity(new_gene, seq):
            num_ops += recurse(results, events, i + 1, new_gene, seq, params)
            if num_ops > 10000:
                marker = typed.List.empty_list(GeneEventNumbaType)
                marker.append(event)
                results.append(marker)  # marker for too many ops
                break

        # handle intron end (exon start)
        if gene[-1].type == EXON_END:
            intron_size = event.pos - gene[-1].pos
            if intron_size > params.max_intron_size:
                break
            if event.type == EXON_START and intron_size >= params.min_intron_size:
                num_ops += recurse(results, events, i + 1, new_gene, seq, params)

    return num_ops


@njit
def copy_and_append_gene_event_numba(gene_def: list[GeneEventNumbaType], event: GeneEventNumbaType) -> list[GeneEventNumbaType]:
    """ Helper function to copy and append a GeneEvent to a list of GeneEvents """
    # numba doesn't support list.append, so we have to create a new list
    new_gene_def = typed.List.empty_list(GeneEventNumbaType)
    for e in gene_def:
        new_gene_def.append(e)
    #     print(f'{type(e)=}')
    # print(f'{type(event)=}')
    new_gene_def.append(event)
    return new_gene_def


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
    event_consistency_score = 0
    if num_vals:
        event_consistency_score = (summed_scores / num_vals + 1) / 2 # scale from range -1,1 to range 0,1

    cds_seq = build_cds_seq(seq, gene_call)
    cds_is_multiple_of_three = len(cds_seq) % 3 == 0
    num_stop_codons = count_stop_codons(cds_seq)
    seq_ends_with_stop_codon = ends_with_stop_codon(cds_seq)

    # cds start and end, exon start and end--the more/higher the more confident we are
    cds_ends_score = 0
    for e in (gene_call[0], gene_call[-1]):
        cds_ends_score += preds[EVENT_TYPE_MAP[e.type]][e.pos] / 2 # scale from range 0,2 to range 0,1

    gene_length_score = (gene_call[-1].pos - gene_call[0].pos) / 10000  # slight preference for longer genes

    score = (
        event_consistency_score +
        cds_ends_score +
        gene_length_score +
        (-np.inf if not cds_is_multiple_of_three else 0) +
        (-np.inf if num_stop_codons > 1 else 0) +
        (-np.inf if not seq_ends_with_stop_codon else 0)
    )

    return score


@njit
def build_cds_seq(seq: str, gene_call: list[GeneEvent]):
    last_pos = None
    cds_seq_list = []
    for pos, typ, score in gene_call:
        if typ in (CDS_START, EXON_START):
            last_pos = pos
        elif typ in (CDS_END, EXON_END) and last_pos is not None:
            # gene_ml cds_end and exon_end predictions have off by one issue
            cds_seq_list.append(seq[last_pos:pos+1])
            last_pos = pos+1
    return ''.join(cds_seq_list)


@njit
def filter_events_for_one_gene(events: list[GeneEvent]) -> list[GeneEvent]:
    """ Filter events for one gene so we limit the number of events considered, as it impacts recursion depth and
    performance
    """

    # region_size = events[-1].pos - events[0].pos
    # max_num_event_per_type = region_size // 200  # maybe an intron every 400 bp, and allow two possibilities for each
    # max_num_event_per_type = max_num_event_per_type * 2 + 5  # scale up
    num_intron_events = 0
    for e in events:
        if e.type in (EXON_START, EXON_END) and e.score > 0.1:
            num_intron_events += 1
    max_num_event_per_type = int(num_intron_events)  # * 2/3)

    event_subset = typed.List.empty_list(GeneEventNumbaType)

    # keep just the first cds_start event
    assert events[0].type == CDS_START, 'events must start with a cds_start event'
    event_subset.append(events[0])

    for event_type in (CDS_END, EXON_START, EXON_END):
        events_by_type = typed.List.empty_list(GeneEventNumbaType)
        for event in events:
            if event.type == event_type:
                events_by_type.append(event)

        if event_type == CDS_END:
            # keep all CDS_END events since they don't impact recursion
            event_subset.extend(events_by_type)
        else:
            events_by_type.sort(key=lambda x: x.score, reverse=True)
            event_subset.extend(events_by_type[:max_num_event_per_type])

    event_subset.sort()  # sort by pos

    return event_subset


@njit
def produce_gene_calls(preds: np.ndarray, events: list[GeneEvent], seq: str, contig_id: str, logs: list, params: namedtuple) -> list[tuple[float, list[GeneEvent]]]:
    """ for a given set of events corresponding to a contig / candidate gene region, produce all possible gene calls"""
    function_start_time = python_time()
    start_time = python_time()
    last_end_idx = -1
    skip_till_next_end_idx = False
    all_best_scores = []
    for start_idx in range(len(events)):
        start_event = events[start_idx]

        if start_event.type == CDS_START:
            end_idx = get_end_idx(start_idx, events, preds)
            if end_idx - start_idx < 2:
                continue

            cds_end_found = False
            for e in events[start_idx:end_idx+1]:
                if e.type == CDS_END:
                    cds_end_found = True
                    if end_idx != last_end_idx:
                        last_end_idx = end_idx
                        start_time = python_time()
                        skip_till_next_end_idx = False
                    break
            if not cds_end_found:
                continue
            if skip_till_next_end_idx:
                continue

            one_gene_events = filter_events_for_one_gene(events[start_idx:end_idx+1])
            # one_gene_events = events[start_idx:end_idx+1]
            gene_calls = typed.List.empty_list(GeneCallNumbaType)
            recurse_start_time = python_time()
            recurse(gene_calls, one_gene_events, 0, typed.List.empty_list(GeneEventNumbaType), seq, params)

            if params.debug:
                print(start_idx, end_idx, prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx]),
                      ';', len(gene_calls), 'gene calls',
                      )

            if gene_calls and len(gene_calls[-1]) == 1:
                # this is a marker for too many ops
                elapsed = python_time() - recurse_start_time
                with objmode(log='unicode_type'):
                    log = ' '.join([
                        'too many ops for', contig_id, str(start_idx), str(end_idx),
                        prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx]),
                        prettify_gene_event(gene_calls[-1][0]),
                        '; elapsed time:', str(np.round(elapsed, 2))])
                logs.append(log)

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

            if params.gene_range_time_limit and python_time() - start_time > params.gene_range_time_limit:
                log = ' '.join(['time limit exceeded for', contig_id, str(start_idx), str(end_idx),
                                prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx])])
                logs.append(log)

                # short circuit this gene region
                skip_till_next_end_idx = True
    all_best_scores.sort(reverse=True)

    elapsed = python_time() - function_start_time
    if elapsed > 600:
        with objmode(log='unicode_type'):
            log = 'slow {contig_id} at {elapsed:.2f}s'.format(contig_id=contig_id, elapsed=elapsed)
        logs.append(log)

    return all_best_scores


def get_compact_events(events: list[GeneEvent]):

    def short_score(score):
        if score >= 0.95:
            return '1'
        return f'{score:.1f}'[-2:]

    event_type_char_map = {CDS_START: 'S', CDS_END: 'E', EXON_START: 's', EXON_END: 'e'}

    return ', '.join(f'{c.pos}{event_type_char_map[c.type]}:{short_score(c.score)}' for c in events)


def get_gene_range(gene_call: list[GeneEvent], is_rc: bool, sequence_length: int) -> tuple[int, int]:
    if is_rc:
        start_pos = sequence_length - gene_call[-1].pos
        end_pos = sequence_length - gene_call[0].pos
    else:
        start_pos = gene_call[0].pos
        end_pos = gene_call[-1].pos

    return start_pos, end_pos


def filter_best_scored_gene_calls(sequence_length: int, all_best_scores: list[tuple[float, list[GeneEvent]]],
                                  rc_all_best_scores: list[tuple[float, list[GeneEvent]]]):
    """ This takes potential gene calls on both forward strand and reverse complement, filters them to remove
    overlapping calls, starting from highest scores first, and returns the best ones"""

    # setup reverse complement calls if applicable
    all_best_scores = [(score, gene_call, False) for score, gene_call in all_best_scores]
    if rc_all_best_scores is not None:
        all_best_scores += [(score, gene_call, True) for score, gene_call in rc_all_best_scores]
    all_best_scores.sort(reverse=True)

    seen = np.zeros(sequence_length, dtype='int8')
    filtered_best_scores = []
    for score, gene_call, is_rc in all_best_scores:
        good = True
        if score < 1:
            # minimum score threshold: this will filter out small genes with little is_exon score support
            good = False

        start_pos, end_pos = get_gene_range(gene_call, is_rc, sequence_length)

        if not good or np.any(seen[start_pos:end_pos]):
            # if low score or overlaps with an already seen gene call, skip it
            continue
        filtered_best_scores.append([score, gene_call, is_rc])
        seen[start_pos:end_pos] = 1 if not is_rc else 2

    return filtered_best_scores, seen


def build_coords(gene_call: list[GeneEvent], offset: int, width: int, reverse_complement: bool) -> tuple[int, int, str]:
    if not reverse_complement:
        return offset+gene_call[0].pos, offset+gene_call[-1].pos, '+'
    else:
        return offset+(width-gene_call[-1].pos), offset+(width-gene_call[0].pos), '-'


def run_model(model: ResidualModelBase, seq: str, forward_strand_only=False) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Build gene calls from a sequence using the GeneML model. Note that the coordinates in filtered_scored_gene_calls are
    relative to the sequence and the strand, so they are not absolute coordinates in the genome or even of the input
    sequence. See build_coords for converting to genomic absolute coordinates.
    """

    seq = seq.upper()

    preds = chunked_seq_predict(model, seq)
    if not forward_strand_only:
        rc_seq = reverse_complement(seq)
        rc_preds = chunked_seq_predict(model, rc_seq)
    else:
        rc_seq = None
        rc_preds = None

    return preds, rc_preds, seq, rc_seq


def build_gene_calls(preds: np.ndarray, rc_preds: np.ndarray, seq: str, rc_seq: str, contig_id: str, params: namedtuple):
    if params.debug:
        print('\n******************** forward strand')
    events = get_gene_ml_events(preds, params)
    logs = typed.List.empty_list(types.unicode_type)
    scored_gene_calls = produce_gene_calls(preds, events, seq, contig_id + ' forward strand', logs, params)

    if not params.forward_strand_only:
        if params.debug:
            print('\n******************** reverse strand')
        rc_events = get_gene_ml_events(rc_preds, params)
        rc_scored_gene_calls = produce_gene_calls(rc_preds, rc_events, rc_seq, contig_id + ' reverse strand', logs, params)
    else:
        rc_events = None
        rc_scored_gene_calls = None

    # gene calling
    sequence_length = len(seq)
    filtered_scored_gene_calls, seen = filter_best_scored_gene_calls(
        sequence_length, scored_gene_calls, rc_all_best_scores=rc_scored_gene_calls)

    return filtered_scored_gene_calls, logs
