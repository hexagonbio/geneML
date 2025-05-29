import numba
import numpy as np
from gene_ml.gene_caller import GeneEvent, CDS_START, EXON_START, CDS_END, EXON_END
from gene_ml.model_loader import MODEL_IS_EXON, MODEL_IS_INTRON, MODEL_CDS_START, MODEL_CDS_END, MODEL_EXON_START, \
    MODEL_EXON_END
from numba import njit, types, typed, objmode


def get_exon_offsets(gene_call: list[GeneEvent]):
    last_pos = None
    for pos, typ, score in gene_call:
        if typ in (CDS_START, EXON_START):
            last_pos = pos
        elif typ in (CDS_END, EXON_END) and last_pos is not None:
            # gene_ml cds_end and exon_end predictions have off by one issue
            yield last_pos, pos
            last_pos = pos+1


def build_gff_coords(chr_name, source, gene_id, gene_call: list[GeneEvent], offset: int, width: int, reverse_complement: bool) -> tuple[int, int, str]:
    gff_rows = []
    # seqname, source, feature, start, end, score, strand, frame, attributes

    # gene record
    if not reverse_complement:
        start, end, strand = offset + gene_call[0].pos + 1, offset + gene_call[-1].pos + 1, '+'
    else:
        start, end, strand = offset + (width - gene_call[-1].pos), offset + (width - gene_call[0].pos), '-'
    gff_rows.append((
        chr_name,
        source,
        "gene",
        start,
        end,
        ".",
        strand,
        ".",
        f"ID={gene_id}",
    ))

    # mRNA record
    gff_rows.append((
        chr_name,
        source,
        "mRNA",
        start,
        end,
        ".",
        strand,
        ".",
        f"ID={gene_id}m;Parent={gene_id}",
    ))

    # exon records
    for i, (start, end) in enumerate(get_exon_offsets(gene_call)):
        if not reverse_complement:
            start, end = offset + start + 1, offset + end + 1
        else:
            start, end = offset + (width - end), offset + (width - start)
        gff_rows.append((
            chr_name,
            source,
            "exon",
            start,
            end,
            ".",
            strand,
            ".",
            f"ID={gene_id}_exon{i+1};Parent={gene_id}m",
        ))

    return gff_rows


@njit
def build_prediction_scores_seg(contig_id: str, preds: np.ndarray, rc_preds: np.ndarray) -> str:
    num_bases: int = preds.shape[1]
    assert preds.shape == rc_preds.shape, f'preds shape {preds.shape} != rc_preds shape {rc_preds.shape}'
    lines = typed.List.empty_list(types.unicode_type)

    name_to_types = {
        'is_exon/is_intron': (MODEL_IS_EXON, MODEL_IS_INTRON),
        'intron starts/ends': (MODEL_EXON_END, MODEL_EXON_START),
        'cds ends': (MODEL_CDS_START, MODEL_CDS_END),
    }
    score_thresholds = {
        'is_exon/is_intron': 0.2,
        'intron starts/ends': 0.01,
        'cds ends': 0.01,
    }
    strands = ['+', '-']
    ps = [preds, rc_preds[:,::-1]]
    for j in range(2):
        strand = strands[j]
        p = ps[j]
        last_score = -100
        last_i = -1
        for name, (typ1, typ2) in name_to_types.items():
            score_threshold = score_thresholds[name]
            for i in range(num_bases):
                score1 = p[typ1, i]
                score2 = p[typ2, i]
                score = round(numba.float64(score1 if score1 > score2 else -score2), 2)
                if abs(score) < score_threshold:
                    score = 0
                if score != last_score:
                    if last_i >= 0:
                        with objmode(row="unicode_type"):
                            cols = (name + strand, contig_id, last_i, i, last_score)
                            row = '\t'.join(map(str, cols))
                        lines.append(row)
                    last_score = score
                    last_i = i

    return '\n'.join(lines)
