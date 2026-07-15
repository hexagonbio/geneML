"""Classify alternative transcripts from a GFF3 file.

This module reconstructs geneML transcript objects from a GFF3 file, selects the
primary transcript per locus by longest CDS length, and labels the remaining
transcripts using the existing transcript-variant classifier.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from geneml.types import EXON_END, EXON_START, Exon, GeneEvent, Transcript, TranscriptVariant


@dataclass
class ParsedTranscript:
    """Transcript data recovered from a GFF3 mRNA feature."""

    transcript_id: str
    parent_id: str
    contig_id: str
    strand: int
    cds_segments: list[tuple[int, int]]
    exon_segments: list[tuple[int, int]]

    def segments_for_classification(self) -> list[tuple[int, int]]:
        """Return the segments used for transcript reconstruction.

        CDS segments are preferred because geneML classifies coding structure.
        Exon segments are used as a fallback for GFF3s that do not provide CDS
        child features.
        """

        if self.cds_segments:
            return self.cds_segments
        return self.exon_segments

    def cds_length(self) -> int:
        """Return the total CDS length in bases."""

        return sum(end - start + 1 for start, end in self.cds_segments)


def parse_gff3_attributes(attribute_text: str) -> list[tuple[str, str]]:
    """Parse a GFF3 attribute column while preserving key order."""

    attributes: list[tuple[str, str]] = []
    if not attribute_text or attribute_text == ".":
        return attributes

    for part in attribute_text.rstrip(";").split(";"):
        part = part.strip()
        if not part:
            continue
        key, separator, value = part.partition("=")
        if not separator:
            continue
        attributes.append((key.strip(), value.strip()))
    return attributes


def format_gff3_attributes(attributes: list[tuple[str, str]]) -> str:
    """Format ordered GFF3 attributes back into a column string."""

    return ";".join(f"{key}={value}" for key, value in attributes)


def replace_attribute(attributes: list[tuple[str, str]], key: str, value: str) -> list[tuple[str, str]]:
    """Replace the first occurrence of an attribute or append it."""

    replaced = False
    new_attributes: list[tuple[str, str]] = []
    for current_key, current_value in attributes:
        if current_key == key and not replaced:
            new_attributes.append((key, value))
            replaced = True
        else:
            new_attributes.append((current_key, current_value))
    if not replaced:
        new_attributes.append((key, value))
    return new_attributes


def parse_strand(strand_text: str) -> int:
    """Convert a GFF3 strand symbol into geneML strand integers."""

    if strand_text == "+":
        return 1
    if strand_text == "-":
        return -1
    raise ValueError(f"Unsupported strand value in GFF3: {strand_text!r}")


def parse_gff3_segment(start_text: str, end_text: str) -> tuple[int, int]:
    """Convert 1-based inclusive GFF3 coordinates into geneML coordinates."""

    start = int(start_text) - 1
    end = int(end_text)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid GFF3 interval: {start_text!r}-{end_text!r}")
    return start, end


def normalize_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort genomic segments into ascending genomic order."""

    return sorted(segments, key=lambda interval: (interval[0], interval[1]))


def load_gff3_transcripts(lines: list[str]) -> tuple[list[str], dict[str, ParsedTranscript], dict[str, list[ParsedTranscript]]]:
    """Parse GFF3 lines into transcript records grouped by locus."""

    transcripts_by_id: dict[str, ParsedTranscript] = {}
    transcripts_by_locus: dict[str, list[ParsedTranscript]] = defaultdict(list)
    cds_segments_by_transcript: dict[str, list[tuple[int, int]]] = defaultdict(list)
    exon_segments_by_transcript: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for raw_line in lines:
        if not raw_line.strip() or raw_line.startswith("#"):
            continue

        columns = raw_line.rstrip("\n").split("\t")
        if len(columns) != 9:
            continue

        feature_type = columns[2]
        if feature_type not in {"gene", "mRNA", "CDS", "exon"}:
            continue

        attributes = dict(parse_gff3_attributes(columns[8]))

        if feature_type == "mRNA":
            transcript_id = attributes.get("ID")
            if not transcript_id:
                raise ValueError("Encountered an mRNA feature without an ID attribute")

            parent = attributes.get("Parent")
            if parent is None:
                parent = attributes["gene_id"]
            parent_id = parent.split(",")[0]

            transcript = ParsedTranscript(
                transcript_id=transcript_id,
                parent_id=parent_id,
                contig_id=columns[0],
                strand=parse_strand(columns[6]),
                cds_segments=[],
                exon_segments=[],
            )
            transcripts_by_id[transcript_id] = transcript
            transcripts_by_locus[parent_id].append(transcript)

        elif feature_type in {"CDS", "exon"}:
            parent_text = attributes.get("Parent")
            if not parent_text:
                continue

            segment = parse_gff3_segment(columns[3], columns[4])
            for parent_id in parent_text.split(","):
                if feature_type == "CDS":
                    cds_segments_by_transcript[parent_id].append(segment)
                else:
                    exon_segments_by_transcript[parent_id].append(segment)

    for transcript_id, transcript in transcripts_by_id.items():
        transcript.cds_segments = normalize_segments(cds_segments_by_transcript.get(transcript_id, []))
        transcript.exon_segments = normalize_segments(exon_segments_by_transcript.get(transcript_id, []))

    return lines, transcripts_by_id, transcripts_by_locus


def build_transcript_model(transcript: ParsedTranscript, group_id: int) -> Transcript:
    """Create a geneML Transcript object from a parsed GFF3 transcript."""

    segments = transcript.segments_for_classification()
    if not segments:
        raise ValueError(f"Transcript {transcript.transcript_id} has no CDS or exon segments")

    exons: list[Exon] = []
    events: list[GeneEvent] = []
    frame = 0
    for start, end in segments:
        phase = (3 - frame) % 3
        exon = Exon(
            start=start,
            end=end,
            events=(
                GeneEvent(start, EXON_START, 0.0),
                GeneEvent(end - 1, EXON_END, 0.0),
            ),
            score=0.0,
            phase=phase,
        )
        exons.append(exon)
        events.extend(exon.events)
        frame = (frame + (end - start)) % 3

    return Transcript(
        start=segments[0][0],
        end=segments[-1][1],
        strand=transcript.strand,
        events=tuple(events),
        score=0.0,
        exons=tuple(exons),
        group_id=group_id,
        transcript_id=transcript.transcript_id,
    )


def classify_transcripts(transcripts_by_locus: dict[str, list[ParsedTranscript]]) -> dict[str, TranscriptVariant]:
    """Classify all transcripts in the file and return transcript ID to variant."""

    variants: dict[str, TranscriptVariant] = {}

    for group_id, transcripts in enumerate(transcripts_by_locus.values()):
        if not transcripts:
            continue

        strands = {transcript.strand for transcript in transcripts}
        if len(strands) != 1:
            locus_id = transcripts[0].parent_id
            raise ValueError(f"Locus {locus_id} contains transcripts on multiple strands")

        primary_record = max(transcripts, key=lambda transcript: transcript.cds_length())

        primary = build_transcript_model(primary_record, group_id)
        primary.set_transcript_variant(TranscriptVariant.PRIMARY)
        variants[primary.transcript_id] = TranscriptVariant.PRIMARY

        for transcript_record in transcripts:
            if transcript_record.transcript_id == primary_record.transcript_id:
                continue
            transcript = build_transcript_model(transcript_record, group_id)
            if not transcript.overlaps_with(primary):
                transcript.set_transcript_variant(TranscriptVariant.UNKNOWN)
                variants[transcript.transcript_id] = TranscriptVariant.UNKNOWN
                continue
            variant = transcript.classify_transcript_variant(primary)
            variants[transcript.transcript_id] = variant


    return variants


def rewrite_gff3_lines(lines: list[str], variants_by_transcript_id: dict[str, TranscriptVariant]) -> list[str]:
    """Rewrite mRNA features with TranscriptVariant attributes."""

    rewritten_lines: list[str] = []
    for raw_line in lines:
        if not raw_line.strip() or raw_line.startswith("#"):
            rewritten_lines.append(raw_line)
            continue

        columns = raw_line.rstrip("\n").split("\t")
        if len(columns) != 9 or columns[2] != "mRNA":
            rewritten_lines.append(raw_line)
            continue

        attributes = parse_gff3_attributes(columns[8])
        attribute_map = dict(attributes)
        transcript_id = attribute_map.get("ID")
        if transcript_id and transcript_id in variants_by_transcript_id:
            attributes = replace_attribute(attributes, "TranscriptVariant", variants_by_transcript_id[transcript_id].name)
            columns[8] = format_gff3_attributes(attributes)
            rewritten_lines.append("\t".join(columns) + "\n")
            continue

        rewritten_lines.append(raw_line)

    return rewritten_lines


def classify_gff3_text(gff3_text: str) -> str:
    """Classify a GFF3 document and return the rewritten text."""

    lines = gff3_text.splitlines(keepends=True)
    _, transcripts_by_id, transcripts_by_locus = load_gff3_transcripts(lines)
    if not transcripts_by_id:
        return gff3_text

    variants_by_transcript_id = classify_transcripts(transcripts_by_locus)
    rewritten_lines = rewrite_gff3_lines(lines, variants_by_transcript_id)
    return "".join(rewritten_lines)


def classify_gff3_file(input_path: str, output_path: str) -> None:
    """Classify a GFF3 file and write the rewritten annotations."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    rewritten_text = classify_gff3_text(input_file.read_text(encoding="utf-8"))
    output_file.write_text(rewritten_text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the classification wrapper."""

    parser = argparse.ArgumentParser(description="Classify transcripts in a GFF3 file using geneML splicing labels.")
    parser.add_argument("input", help="Input GFF3 file")
    parser.add_argument("output", help="Output GFF3 file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for GFF3 transcript classification."""

    args = parse_args(argv)
    classify_gff3_file(args.input, args.output)


if __name__ == "__main__":
    main()
