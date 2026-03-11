import logging

from geneml.types import Exon, SplicingType, Transcript

logger = logging.getLogger("geneml")

def get_ordered_introns(exons: list[Exon] | tuple[Exon, ...], strand: int) -> list[tuple[int,int]]:
    """
    Returns introns (donor, acceptor) in transcriptional order.

    Args:
        exons: List or tuple of Exon objects, sorted by genomic position.
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
    assert primary.exons != alt.exons, (
        f"Identical transcripts: {primary.transcript_id} and {alt.transcript_id}"
    )

    events = set()
    strand = primary.strand

    # Order by transcriptional order
    P = get_ordered_introns(primary.exons, strand)
    A = get_ordered_introns(alt.exons, strand)

    # Only compare introns within the shared genomic region of the two transcripts
    shared_range = (max(primary.start, alt.start), min(primary.end, alt.end))

    assert shared_range[0] < shared_range[1], (
            "Alternative transcript %s (%s-%s) does not overlap with "
            "primary transcript %s (%s-%s)",
            alt.transcript_id, alt.start, alt.end,
            primary.transcript_id, primary.start, primary.end,
        )

    P = get_introns_in_range(P, shared_range, strand)
    A = get_introns_in_range(A, shared_range, strand)

    P_set = set(P)
    A_set = set(A)

    # Track junctions consumed by exon skipping to avoid double-counting them
    consumed_P = set()
    consumed_A = set()

    # 1. ALTERNATIVE FIRST / LAST EXON
    if strand == 1:
        primary_first, alt_first = primary.start, alt.start
        primary_last, alt_last = primary.end, alt.end
    elif strand == -1:
        primary_first, alt_first = primary.end, alt.end
        primary_last, alt_last = primary.start, alt.start
    else:
        raise ValueError(f"Invalid strand: {strand}")

    if alt_first != primary_first:
        events.add(SplicingType.ALT_FIRST_EXON)
    if alt_last != primary_last:
        events.add(SplicingType.ALT_LAST_EXON)

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
                # Skip if this is in a terminal exon already recognized as alt event
                at_alt_first_exon = (
                    ((s1, e1) == P[0] or (s2, e2) == A[0])
                    and SplicingType.ALT_FIRST_EXON in events
                )
                at_alt_last_exon = (
                    ((s1, e1) == P[-1] or (s2, e2) == A[-1])
                    and SplicingType.ALT_LAST_EXON in events
                )
                if not (at_alt_first_exon or at_alt_last_exon):
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

    assert events, (
        f"No splicing events detected between {primary.transcript_id} and {alt.transcript_id}; "
        f"this should not happen"
    )

    if len(events) == 1:
        return events.pop()
    return SplicingType.COMPLEX
