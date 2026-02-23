from geneml.types import SplicingType, Transcript


def get_ordered_introns(exons: list, strand: int) -> list[tuple[int,int]]:
    """
    Returns introns (donor, acceptor) in transcriptional order.

    Args:
        exons: list of Exon objects, sorted by genomic position.
        strand: +1 for forward strand, -1 for reverse strand.

    Returns:
        List of introns as (donor, acceptor) tuples in transcriptional order.
    """
    if len(exons) == 1:
        return []  # single-exon transcript has no introns

    introns = []
    for i in range(len(exons) - 1):
        # donor = 5' splice site, acceptor = 3' splice site (transcriptional)
        if strand == 1:
            donor = exons[i].end
            acceptor = exons[i + 1].start
        elif strand == -1:
            donor = exons[i].start
            acceptor = exons[i + 1].end
        else:
            raise ValueError(f"Invalid strand: {strand}")
        introns.append((donor, acceptor))
    return introns


def get_introns_in_range(introns: list[tuple[int,int]], range: tuple[int,int], strand: int
                         ) -> list[tuple[int,int]]:
    """Returns introns that fall within the specified genomic range.

    Args:
        introns: List of intron tuples (donor, acceptor) in transcriptional order.
        range: Tuple of (start, end) genomic coordinates defining the range.
        strand: +1 for forward strand, -1 for reverse strand.

    Returns:
        List of introns that fall within the specified range.
    """
    if strand == 1:
        return [(s, e) for s, e in introns if s >= range[0] and e <= range[1]]
    elif strand == -1:
        return [(s, e) for s, e in introns if s <= range[1] and e >= range[0]]
    else:
        raise ValueError(f"Invalid strand: {strand}")


def get_alternative_splicing_type(primary: Transcript, alt: Transcript) -> SplicingType:
    """Classify alternative splicing events between a primary and alternative transcript.

    Compares intron junctions and terminal exon boundaries to detect exon skipping,
    alternative 3' and 5' splice sites, intron retention, and alternative first/last
    exons. If multiple event types are detected, the transcript is labeled as
    complex.

    Args:
        primary: The reference transcript to compare against.
        alt: The alternative transcript being classified.

    Returns:
        The assigned SplicingType for the alternative transcript.
    """
    assert primary is not alt, "Should not compare to self"

    events = set()
    strand = primary.strand

    # Order by transcriptional order
    P = get_ordered_introns(primary.exons, strand)
    A = get_ordered_introns(alt.exons, strand)

    # Only compare introns within the shared genomic region of the two transcripts
    shared_range = (max(primary.start, alt.start), min(primary.end, alt.end))
    P = get_introns_in_range(P, shared_range, strand)
    A = get_introns_in_range(A, shared_range, strand)

    P_set = set(P)
    A_set = set(A)

    # Track junctions consumed by exon skipping to avoid double-counting them
    consumed_P = set()
    consumed_A = set()

    # 1. ALTERNATIVE FIRST / LAST EXON
    if strand == 1:
        if alt.exons[0].start != primary.exons[0].start:
            events.add(SplicingType.ALT_FIRST_EXON)
        if alt.exons[-1].end != primary.exons[-1].end:
            events.add(SplicingType.ALT_LAST_EXON)
    elif strand == -1:
        if alt.exons[0].end != primary.exons[0].end:
            events.add(SplicingType.ALT_FIRST_EXON)
        if alt.exons[-1].start != primary.exons[-1].start:
            events.add(SplicingType.ALT_LAST_EXON)
    else:
        raise ValueError(f"Invalid strand: {strand}")

    # 2. EXON SKIPPING
    for s, e in P:
        for i in range(len(A) - 1):
            s1, e1 = A[i]
            s2, e2 = A[i + 1]

            if s1 == s and e2 == e:
                events.add(SplicingType.EXON_SKIPPING)
                consumed_P.add((s, e))
                consumed_A.add((s1, e1))
                consumed_A.add((s2, e2))

    for s, e in A:
        for i in range(len(P) - 1):
            s1, e1 = P[i]
            s2, e2 = P[i + 1]

            if s1 == s and e2 == e:
                events.add(SplicingType.EXON_SKIPPING)
                consumed_A.add((s, e))
                consumed_P.add((s1, e1))
                consumed_P.add((s2, e2))

    # 3. ALTERNATIVE 3' / 5' SPLICE SITES
    for s1, e1 in P:
        if (s1, e1) in consumed_P:
            continue
        for s2, e2 in A:
            if (s2, e2) in consumed_A:
                continue
            if s1 == s2 and e1 != e2:
                # Skip if this is the terminal exon boundary (already counted as ALT_LAST_EXON)
                if (s1, e1) == P[-1] or (s2, e2) == A[-1] and SplicingType.ALT_LAST_EXON in events:
                    pass
                else:
                    events.add(SplicingType.ALT_3_SPLICE_SITE)
                consumed_P.add((s1, e1))
                consumed_A.add((s2, e2))
            if e1 == e2 and s1 != s2:
                events.add(SplicingType.ALT_5_SPLICE_SITE)
                consumed_P.add((s1, e1))
                consumed_A.add((s2, e2))

    # 4. INTRON RETENTION
    remaining_P = (P_set - A_set) - consumed_P
    remaining_A = (A_set - P_set) - consumed_A

    if remaining_P or remaining_A:
        events.add(SplicingType.INTRON_RETENTION)


    # Assign final splicing type
    if not events:
        return SplicingType.UNKNOWN
    elif len(events) == 1:
        return events.pop()
    return SplicingType.COMPLEX
