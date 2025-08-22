import os

import numba
import numpy as np
from Bio.Seq import reverse_complement, translate
from numba import njit, objmode, typed, types

from geneml.gene_caller import CDS_END, CDS_START, EXON_END, EXON_START, GeneEvent, build_cds_seq
from geneml.model_loader import (
    MODEL_CDS_END,
    MODEL_CDS_START,
    MODEL_EXON_END,
    MODEL_EXON_START,
    MODEL_IS_EXON,
    MODEL_IS_INTRON,
)


def get_exon_offsets(gene_call: list[GeneEvent]):
    last_pos = None
    for pos, typ, score in gene_call:
        if typ in (CDS_START, EXON_START):
            last_pos = pos
        elif typ in (CDS_END, EXON_END) and last_pos is not None:
            # gene_ml cds_end and exon_end predictions have off by one issue
            yield last_pos, pos
            last_pos = pos+1


def build_gff_coords(chr_name, source, gene_id, gene_call: list[GeneEvent], offset: int, width: int, reverse_complement: bool) -> list:
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
        f"ID={gene_id}_mRNA;Parent={gene_id}",
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
            f"ID={gene_id}_exon{i+1};Parent={gene_id}_mRNA",
        ))
        gff_rows.append((
            chr_name,
            source,
            "CDS",
            start,
            end,
            ".",
            strand,
            ".",
            f"ID={gene_id}_CDS;Parent={gene_id}_mRNA",
        ))

    return gff_rows


def contig_gene_generator(contigs: dict[str, str], results: dict[str, list]):
    gene_count = 0
    for contig_id, seq in contigs.items():
        if contig_id not in results:
            continue
        filtered_scored_gene_calls = results[contig_id]
        filtered_scored_gene_calls.sort(key=lambda x: x[1][0].pos)  # sort by gene position
        contig_length = len(seq)
        for gene_info in filtered_scored_gene_calls:
            gene_count += 1
            score, gene_call, is_rc = gene_info
            gene_id = f'GML{gene_count:05d}'

            yield contig_id, gene_id, is_rc, gene_call, contig_length


def write_gff_file(contigs: dict[str, str], results: dict[str, list], outpath: str):
    gff_version = "3.1.26"
    gff_header = ' '.join(['##gff-version', gff_version])
    all_gff_rows = [(gff_header,)]
    for contig_id, seq in contigs.items():
        region_header = ' '.join(['##sequence-region', contig_id, '1', str(len(seq))])
        all_gff_rows.append((region_header,))

    for contig_id, gene_id, is_rc, gene_call, contig_length in contig_gene_generator(contigs, results):
        all_gff_rows.extend(build_gff_coords(contig_id, 'geneML', gene_id, gene_call, 0, contig_length, is_rc))

    if dirname := os.path.dirname(outpath):
        os.makedirs(dirname, exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        for gff_row in all_gff_rows:
            formatted_gff_row = '\t'.join(map(str, gff_row))
            f.write(f'{formatted_gff_row}\n')


def write_fastas(contigs: dict[str, str], results: dict[str, list], basepath: str):
    line_length = 60
    rc_contigs = {contig_id: reverse_complement(seq) for contig_id, seq in contigs.items()}
    cds_filename = basepath+'.cds.fasta'
    prot_filename = basepath+'.prot.fasta'
    with open(cds_filename, 'w') as f_cds, open(prot_filename, 'w') as f_prot:
        for contig_id, gene_id, is_rc, gene_call, contig_length in contig_gene_generator(contigs, results):
            contig_seq = contigs[contig_id] if not is_rc else rc_contigs[contig_id]

            cds_seq = build_cds_seq(contig_seq, gene_call)
            print(f'>{gene_id}', file=f_cds)
            for i in range(0, len(cds_seq), line_length):
                print(cds_seq[i:i+line_length], file=f_cds)

            aa_seq = translate(cds_seq)
            print(f'>{gene_id}', file=f_prot)
            for i in range(0, len(aa_seq), line_length):
                print(aa_seq[i:i+line_length], file=f_prot)


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
