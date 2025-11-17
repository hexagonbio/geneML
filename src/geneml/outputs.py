import os

import numba
import numpy as np
from Bio.Seq import reverse_complement, translate
from numba import njit, objmode, typed, types

from geneml.model_loader import (
    MODEL_CDS_END,
    MODEL_CDS_START,
    MODEL_EXON_END,
    MODEL_EXON_START,
    MODEL_IS_EXON,
    MODEL_IS_INTRON,
)
from geneml.types import Gene


def build_gff_coords(contig_id: str, gene: Gene, source: str, offset: int = 0) -> list:
    gff_rows = []
    gene_id = gene.gene_id
    strand = '+' if gene.strand == 1 else '-'

    def get_start_coordinate(start_value, offset=offset):
        return offset + start_value + 1     # GFF is 1-based
    def get_end_coordinate(end_value, offset=offset):
        return offset + end_value

    # seqname, source, feature, start, end, score, strand, frame, attributes
    # gene record
    gff_rows.append((
        contig_id,
        source,
        "gene",
        get_start_coordinate(gene.start),
        get_end_coordinate(gene.end),
        ".",
        strand,
        ".",
        f"ID={gene_id}",
    ))

    for transcript in gene.transcripts:
        transcript_id = transcript.transcript_id
        # mRNA record
        gff_rows.append((
            contig_id,
            source,
            "mRNA",
            get_start_coordinate(transcript.start),
            get_end_coordinate(transcript.end),
            f"{transcript.score:.3f}",
            strand,
            ".",
            f"ID={transcript_id};Parent={gene_id}",
        ))

        # exon records
        for i, exon in enumerate(transcript.exons):
            exon_start = get_start_coordinate(exon.start)
            exon_end = get_end_coordinate(exon.end)
            gff_rows.append((
                contig_id,
                source,
                "exon",
                exon_start,
                exon_end,
                ".",
                strand,
                ".",
                f"ID={transcript_id}_exon{i+1};Parent={transcript_id}",
            ))
            gff_rows.append((
                contig_id,
                source,
                "CDS",
                exon_start,
                exon_end,
                ".",
                strand,
                ".",
                f"ID={transcript_id}_CDS{i+1};Parent={transcript_id}",
            ))

    return gff_rows


def write_gff_file(contigs: dict[str, str], results: dict[str, list[Gene]], outpath: str):
    gff_version = "3.1.26"
    gff_header = ' '.join(['##gff-version', gff_version])
    all_gff_rows = [(gff_header,)]

    last_contig_id = None
    for contig_id, genes in results.items():
        if contig_id != last_contig_id:
            last_contig_id = contig_id
            seq = contigs[contig_id]
            region_header = ' '.join(['##sequence-region', contig_id, '1', str(len(seq))])
            all_gff_rows.append((region_header,))
        for gene in genes:
            all_gff_rows.extend(build_gff_coords(contig_id, gene, source='geneML'))

    if dirname := os.path.dirname(outpath):
        os.makedirs(dirname, exist_ok=True)
    with open(outpath, 'w', encoding='utf-8') as f:
        for gff_row in all_gff_rows:
            formatted_gff_row = '\t'.join(map(str, gff_row))
            f.write(f'{formatted_gff_row}\n')


def build_cds_sequences(contigs: dict[str, str], results: dict[str, list[Gene]]
                        ) -> dict[str, str]:
    cdses_by_transcript = {}
    for contig_id, genes in results.items():
        if not genes:
            continue
        seq = contigs[contig_id]
        for gene in genes:
            for transcript in gene.transcripts:
                cds_seq = ''
                # For reverse strand, exons are in genomic order
                # but we need them in transcription order
                exons = transcript.exons if gene.strand == 1 else transcript.exons[::-1]

                for exon in exons:
                    exon_seq = seq[exon.start:exon.end]
                    cds_seq += exon_seq

                if gene.strand == -1:
                    cds_seq = reverse_complement(cds_seq)

                cdses_by_transcript[transcript.transcript_id] = cds_seq
    return cdses_by_transcript


def write_fasta(cdses_by_transcript: dict[str, str], path: str, sequence_type: str):
    if sequence_type == 'cds':
        to_write = cdses_by_transcript
    elif sequence_type == 'protein':
        to_write = {id: translate(seq, cds=True) for id, seq in cdses_by_transcript.items()}
    else:
        raise ValueError(f"Invalid sequence type: {sequence_type}")

    with open(path, 'w', encoding='utf-8') as s:
        for gene_id, seq in to_write.items():
            s.write(f'>{gene_id}\n{seq}\n')


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
