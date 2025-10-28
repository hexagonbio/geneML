import logging
from collections import defaultdict

from geneml.types import Exon, Gene, GeneEvent, Transcript, CDS_END, CDS_START, EXON_END, EXON_START

logger = logging.getLogger("geneml")


def filter_gene_events(gene_events: list[GeneEvent]) -> list[GeneEvent]:
    if not gene_events:
        return []
    filtered_events = []
    for i, event in enumerate(gene_events):
        # skip exon end if followed by cds end
        if (i < len(gene_events) - 1 and
            event.type == EXON_END and
            gene_events[i + 1].type == CDS_END):
            continue
        filtered_events.append(event)
    return filtered_events


def create_exons(gene_events: list[GeneEvent]) -> list[Exon]:
    if not gene_events:
        return []
    assert len(gene_events) % 2 == 0, 'There should be an even number of gene events.'
    exons = []
    for event, next_event in zip(gene_events[::2], gene_events[1::2]):
        if not (event.type in (CDS_START, EXON_START) and next_event.type in (CDS_END, EXON_END)):
            raise ValueError(f'Invalid gene events pair (expected start event and end event): {event}, {next_event}')
        exon = Exon(
            start=event.pos,
            end=next_event.pos + 1,  # end is exclusive
            events=(event, next_event)
        )
        exons.append(exon)
    return exons


def create_transcripts(scored_gene_calls: list[tuple[float, list[GeneEvent]]],
                       contig_length: int, is_rc: bool) -> list[Transcript]:
    transcripts = []
    strand = -1 if is_rc else 1
    for score, gene_events in scored_gene_calls:
        filtered_events = filter_gene_events(gene_events)
        if is_rc:
            start_pos = contig_length - filtered_events[-1].pos
            end_pos = contig_length - filtered_events[0].pos
        else:
            start_pos = filtered_events[0].pos
            end_pos = filtered_events[-1].pos
        transcript = Transcript(
            start=start_pos,
            end=end_pos + 1,    #end is exclusive
            strand=strand,
            events=tuple(filtered_events),
            score=score,
            exons=tuple(create_exons(filtered_events)),
        )
        transcripts.append(transcript)
    return transcripts


def assign_transcripts_to_genes(transcripts_by_contig_id: dict[str, list[Transcript]]
                                ) -> dict[str, list[Gene]]:
    genes_by_contig = defaultdict(list)
    gene_count = 0
    for contig_id, transcripts in transcripts_by_contig_id.items():
        for transcript in transcripts:
            gene_count += 1
            gene_id = f'GML{gene_count:05d}'
            gene = Gene(
                gene_id=gene_id,
                start=transcript.start,
                end=transcript.end,
                strand=transcript.strand,
                transcripts=(transcript,)
            )
            genes_by_contig[contig_id].append(gene)
    return genes_by_contig
