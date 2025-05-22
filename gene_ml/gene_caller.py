import time
from collections import namedtuple

import numpy as np
from numba import njit, typeof, typed, objmode, types

from gene_ml.model_loader import ResidualModelBase, MODEL_CDS_START, MODEL_CDS_END, MODEL_EXON_START, MODEL_EXON_END, \
    MODEL_IS_EXON, MODEL_IS_INTRON
from gene_ml.utils import chunked_seq_predict

# using dataclass would be nice, but numba doesn't support it
GeneEvent = namedtuple('GeneEvent', ['pos', 'type', 'score'])

CDS_START = 0
CDS_END = 1
EXON_START = 2
EXON_END = 3

# EVENT_TYPE_MAP = ('cds_start', 'cds_end', 'exon_start', 'exon_end')
EVENT_TYPE_MAP = (MODEL_CDS_START, MODEL_CDS_END, MODEL_EXON_START, MODEL_EXON_END)

GeneEventNumbaType = typeof(GeneEvent(1, CDS_START, np.float32(0.5)))
GeneCallNumbaType = typeof(typed.List.empty_list(GeneEventNumbaType))

MIN_INTRON_SIZE, MAX_INTRON_SIZE = 30, 400


@njit
def python_time():
    with objmode(out='float64'):
        out = time.time()
    return out


@njit
def prettify_gene_event(event: GeneEvent) -> str:
    with objmode(out='unicode_type'):
        out = '{pos}:{type}:{score:.1f}'.format(pos=event.pos, type=event.type, score=event.score)
    return out


@njit
def get_gene_ml_events(preds: np.ndarray):
    cds_starts = np.where(preds[MODEL_CDS_START] > 0.01)[0]
    cds_ends = np.where(preds[MODEL_CDS_END] > 0.01)[0]
    exon_starts = np.where(preds[MODEL_EXON_START] > 0.01)[0]
    exon_ends = np.where(preds[MODEL_EXON_END] > 0.01)[0]

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
def recurse(results: list[list[GeneEvent]], events: list[GeneEvent], i: int, gene: list, seq: str, debug=False) -> int:
    """ Recursively attempt to build genes from the events list
    """
    num_ops = 1

    # handle beginning of gene
    if i == 0:
        assert events[0].type == CDS_START, 'events must start with a cds_start event'
        gene.append(events[i])
        i += 1

    while True:
        if (not debug and (len(results) >= 100 or results and len(results[-1]) == 1)
                or debug and len(results) >= 10000):
            # limit the number of results to speed up
            break

        event = events[i]
        new_gene = copy_and_append_gene_event_numba(gene, event)

        # handle cds end
        if gene and event.type == CDS_END and (debug or check_sequence_validity(new_gene, seq)):
            results.append(new_gene)

        if i + 1 >= len(events):
            break  # no more additional events to consider

        # handle intron start (exon end)
        if gene[-1].type in {CDS_START, EXON_START} and event.type == EXON_END and (debug or check_sequence_validity(new_gene, seq)):
            num_ops += recurse(results, events, i + 1, new_gene, seq, debug=debug)
            if num_ops > 10000:
                marker = typed.List.empty_list(GeneEventNumbaType)
                marker.append(event)
                results.append(marker)  # marker for too many ops
                break

        # handle intron end (exon start)
        if gene[-1].type == EXON_END:
            intron_size = event.pos - gene[-1].pos
            if intron_size > MAX_INTRON_SIZE:
                break
            if event.type == EXON_START and intron_size >= MIN_INTRON_SIZE:
                num_ops += recurse(results, events, i + 1, new_gene, seq, debug=debug)

        i += 1

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
def score_gene_call(preds: np.ndarray, gene_call: list[GeneEvent], seq: str, debug=False):
    last_pos = None
    summed_scores = 0
    num_vals = 0
    for pos, event, score in gene_call:
        if last_pos is not None:
            is_exon = event != EXON_START
            key = MODEL_IS_EXON if is_exon else MODEL_IS_INTRON
            summed_scores += np.sum(preds[key, last_pos:pos])
            num_vals += pos - last_pos
        last_pos = pos

    mean_is_exon = summed_scores / num_vals if num_vals else 0
    cds_seq = build_cds_seq(seq, gene_call)
    cds_is_multiple_of_three = len(cds_seq) % 3 == 0
    num_stop_codons = count_stop_codons(cds_seq)
    seq_ends_with_stop_codon = ends_with_stop_codon(cds_seq)

    # cds start and end, exon start and end--the more/higher the more confident we are
    cds_ends_score = 0
    for e in (gene_call[0], gene_call[-1]):
        cds_ends_score += preds[EVENT_TYPE_MAP[e.type]][e.pos]

    # incorporate intron scores
    intron_score = 0
    for i, e in enumerate(gene_call[:-1]):
        if e.type == EXON_END:
            e2 = gene_call[i+1]
            if e2.type != EXON_START:
                continue
            intron_score += np.mean(preds[MODEL_IS_INTRON, e.pos+1:e2.pos] - preds[MODEL_IS_EXON, e.pos+1:e2.pos]) - 1

    gene_length_score = (gene_call[-1].pos - gene_call[0].pos) / 10000  # slight preference for longer genes

    score = (
        mean_is_exon +
        intron_score +
        cds_ends_score +
        gene_length_score +
        (-np.inf if not cds_is_multiple_of_three else 0) +
        (-np.inf if num_stop_codons > 1 else 0) +
        (-np.inf if not seq_ends_with_stop_codon else 0)
    )

    if debug:
        gene_call_str_list = []
        for c in gene_call:
            gene_call_str_list.append(prettify_gene_event(c))
        gene_call_str = ', '.join(gene_call_str_list)
        with objmode(out='unicode_type'):
            out = (
                '{score:.3f} {mean_is_exon:.3f} {cds_ends_score:.3f} num_introns={num_introns}\n'
                'cds_is_multiple_of_three={cds_is_multiple_of_three} num_stop_codons={num_stop_codons} seq_ends_with_stop_codon={seq_ends_with_stop_codon}\n'
                'intron_score={intron_score} cds_ends_score={cds_ends_score} gene_length_score={gene_length_score}\n'
                # 'gene_call: {gene_call_str}\n'
            ).format(
                    score=score, mean_is_exon=mean_is_exon, cds_ends_score=cds_ends_score,
                    num_introns=(len(gene_call) - 2) / 2,
                    cds_is_multiple_of_three=cds_is_multiple_of_three,
                    num_stop_codons=num_stop_codons, seq_ends_with_stop_codon=seq_ends_with_stop_codon,
                    intron_score=intron_score,
                    gene_length_score=gene_length_score,
                    gene_call_str=gene_call_str,
            )
        print(out)
        if gene_call_str == 'your gene call here':
            print(cds_seq)
        # print(cds_seq)

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
def produce_gene_calls(preds: np.ndarray, events: list[GeneEvent], seq: str, contig_id: str, logs: list, time_limit=5, debug=False) -> list[tuple[float, list[GeneEvent]]]:
    """ for a given set of events corresponding to a contig / candidate gene region, produce all possible gene calls"""
    debug_start_pos = None  # *************** set cds start position (0-based) to debug a specific gene call *****************
    # debug_start_pos = 2000
    debug2 = False

    function_start_time = python_time()
    start_time = python_time()
    last_end_idx = -1
    skip_till_next_end_idx = False
    all_best_scores = []
    for start_idx in range(len(events)):
        start_event = events[start_idx]
        if debug_start_pos:
            if start_event.pos != debug_start_pos:
                continue
            debug2 = True

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

            if debug:
                pass
                # print(start_idx, end_idx, prettify_gene_event(events[start_idx]),
                #              prettify_gene_event(events[end_idx]), end='', flush=True)

            # start_time = time.time()

            one_gene_events = filter_events_for_one_gene(events[start_idx:end_idx+1])
            # one_gene_events = events[start_idx:end_idx+1]
            gene_calls = typed.List.empty_list(GeneCallNumbaType)
            recurse(gene_calls, one_gene_events, 0, typed.List.empty_list(GeneEventNumbaType), seq, debug=debug2)

            if debug:
                # elapsed = time.time() - start_time
                print(start_idx, end_idx, prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx]),
                      ';', len(gene_calls), 'gene calls',
                      # 'in {.2f}s'.format(elapsed if elapsed > 0.1 else '')
                      )

            if gene_calls and len(gene_calls[-1]) == 1:
                # this is a marker for too many ops
                log = ' '.join(['too many ops for ', contig_id, str(start_idx), str(end_idx),
                                prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx]),
                                prettify_gene_event(gene_calls[-1][0])])
                logs.append(log)
                gene_calls = gene_calls[:-1]

            if gene_calls:

                scores = []
                for gene_call in gene_calls:
                    scores.append((score_gene_call(preds, gene_call, seq), gene_call))
                scores.sort(reverse=True)

                if debug2:
                    print('gene_calls', len(gene_calls))
                    for s in scores[:10]:
                        gene_call = s[1]
                        print(s)
                        print(score_gene_call(preds, gene_call, seq, debug=True), len(gene_call), gene_call[0],
                              gene_call[-1])

                # choose just the best... revisit?
                all_best_scores.append(scores[0])

            if time_limit and python_time() - start_time > time_limit:
                log = ' '.join(['time limit exceeded for', contig_id, str(start_idx), str(end_idx),
                                prettify_gene_event(events[start_idx]), prettify_gene_event(events[end_idx])])
                logs.append(log)
                skip_till_next_end_idx = True
    all_best_scores.sort(reverse=True)

    elapsed = python_time() - function_start_time
    if elapsed > 60:
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
                                  rc_all_best_scores: list[tuple[float, list[GeneEvent]]], debug=False):
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
        if debug and rc_all_best_scores is None and good:
            # show all good candidate gene calls assuming this is a single gene (no reverse complement provided)
            print(f'{good} {score:.3f} {"-" if is_rc else "+"} {min(start_pos, end_pos)}-{max(start_pos, end_pos)} {end_pos - start_pos}',
                  get_compact_events(gene_call))

        if not good or np.any(seen[start_pos:end_pos]):
            # print(f'FILTEREDOUT {good} {score:.3f} {"-" if is_rc else "+"} {min(start_pos, end_pos)}-{max(start_pos, end_pos)} {end_pos - start_pos}',
            #       get_compact_events(gene_call))
            continue
        filtered_best_scores.append([score, gene_call, is_rc])
        seen[start_pos:end_pos] = 1 if not is_rc else 2

    if debug and rc_all_best_scores:
        print('*** final cluster gene calls ***')
        for score, gene_call, is_rc in filtered_best_scores:
            start_pos, end_pos = get_gene_range(gene_call, is_rc, sequence_length)
            print(f'{score:.3f} {"-" if is_rc else "+"} {start_pos}-{end_pos} {gene_call[-1].pos - gene_call[0].pos}',
                  get_compact_events(gene_call))

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
        # rc_seq = reverse_complement(seq)
        rc_seq = seq[::-1].translate(str.maketrans('ACGT', 'TGCA'))
        rc_preds = chunked_seq_predict(model, rc_seq)
    else:
        rc_seq = None
        rc_preds = None

    return preds, rc_preds, seq, rc_seq


def build_gene_calls(preds: np.ndarray, rc_preds: np.ndarray, seq: str, rc_seq: str, contig_id: str,
                     forward_strand_only=False, debug=False):

    if debug:
        print('\n******************** forward strand')
    events = get_gene_ml_events(preds)
    # for event in events:
    #     if event.type == CDS_END:
    #         print(seq[event.pos-6:event.pos+6], seq[event.pos-2:event.pos+1], event.score, event.pos)
    logs = typed.List.empty_list(types.unicode_type)
    scored_gene_calls = produce_gene_calls(preds, events, seq, contig_id + ' forward strand', logs, debug=debug)

    if not forward_strand_only:
        if debug:
            print('\n******************** reverse strand')
        rc_events = get_gene_ml_events(rc_preds)
        rc_scored_gene_calls = produce_gene_calls(rc_preds, rc_events, rc_seq, contig_id + ' reverse strand', logs, debug=debug)
    else:
        rc_events = None
        rc_scored_gene_calls = None

    # gene calling
    if 0 and debug:
        print('forward good events:', get_compact_events([e for e in events if e.score > 0.5]))
        print('introns:')
        print('starts:  GT      ')
        for event in events:
            if event.type == EXON_END:
                print(f'{event.pos:<5d}', seq[event.pos-2:event.pos+6], f'{event.score:.2f}')
        print('ends:     AG    ')
        for event in events:
            if event.type == EXON_START:
                print(f'{event.pos:<5d}', seq[event.pos-6:event.pos+2], f'{event.score:.2f}')
        if not forward_strand_only:
            print('reverse good events:', get_compact_events([e for e in rc_events if e.score > 0.5]))
    sequence_length = len(seq)
    filtered_scored_gene_calls, seen = filter_best_scored_gene_calls(
        sequence_length, scored_gene_calls, rc_all_best_scores=rc_scored_gene_calls, debug=debug)

    if debug and forward_strand_only and filtered_scored_gene_calls:
        get_intron_site_sequence(seq, filtered_scored_gene_calls[0][1], debug=True)

    return filtered_scored_gene_calls, logs


def get_intron_site_sequence(seq, gene_call, debug=False) -> str:
    lines = []
    for i, event in enumerate(gene_call):
        if event.type == CDS_START:
            lines.append((f'cds start ({event.pos}):', seq[event.pos:event.pos + 3]))
            #              intron  2037- 2086: CATGTGCG ACTCAGCA')
            lines.append(('intron consensus:      GTagt    cAGg  __scores_ size',))
        if event.type == CDS_END:
            lines.append((f'cds end ({event.pos}):', seq[event.pos-2:event.pos+1]))
        if event.type == EXON_END and i < len(gene_call) - 1:
            event2 = gene_call[i+1]
            lines.append((f'intron {event.pos:5d}-{event2.pos:<5d}:',
                          seq[event.pos-2:event.pos + 6], seq[event2.pos-6:event2.pos+2],
                          f'{event.score:.2f} {event2.score:.2f} {event2.pos - event.pos:<4d}'))

    if debug:
        for line in lines:
            print(*line)

    return '\n'.join(' '.join(map(str, line)) for line in lines)
