from gene_ml.gene_caller import GeneEvent, CDS_START, EXON_START, CDS_END, EXON_END


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


