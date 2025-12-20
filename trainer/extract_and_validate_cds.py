#!/usr/bin/env python3
"""
Extract and validate CDS sequences in a single pass.

Reads reference FASTA and splice_table, for each gene:
  1. Fetches full flanked region from genome
  2. Splices exons to get CDS
  3. Validates: start codon, frame, no internal stops, terminal stop codon
  4. Auto-fallback: if nearly all entries lack terminal stops, accept CDS
      without terminal stops (common in some annotations) and report whether
      a stop codon sits immediately downstream.
  5. Writes only valid genes to output splice_table and sequence files

Convention: CDS usually includes stop codon (AUGUSTUS-style), but some
annotations exclude it; this script can auto-detect and adapt.
"""

import argparse
import sys
from collections import defaultdict

from Bio.Seq import reverse_complement
from helperlibs.bio import seqio


def load_fasta_index(fasta_path):
    """
    Load FASTA file into memory as a dict: {seqname -> sequence}.
    """
    sequences = {}
    for record in seqio.parse(fasta_path):
        sequences[record.id] = str(record.seq).upper()
    return sequences


def get_flanked_region(chrom, flank_start, flank_end, strand, sequences):
    """
    Extract flanked region from genome.

    Returns: sequence in forward genomic coordinates (NOT reverse-complemented yet)
    """
    if chrom not in sequences:
        return None

    genome_seq = sequences[chrom]

    # Convert 1-based GFF coordinates to 0-based Python indices
    start_idx = int(flank_start) - 1  # 1-based to 0-based
    end_idx = int(flank_end)          # 1-based inclusive to 0-based exclusive

    if start_idx < 0 or end_idx > len(genome_seq):
        return None

    region = genome_seq[start_idx:end_idx].upper()

    return region


def splice_exons(region, seq_start, jn_start_list, jn_end_list, strand):
    """
    Extract spliced CDS from flanked region using exon boundaries.

    Args:
        region: Flanked genomic region (in forward genomic coordinates, not RC'd)
        seq_start: Genomic coordinate of region[0] (1-based GFF coordinate)
        jn_start_list: List of exon start positions (genomic, in transcript order 5' to 3')
        jn_end_list: List of exon end positions (genomic, in transcript order 5' to 3')
        strand: '+' or '-'

    Returns:
        Spliced CDS sequence (sense strand, will RC if minus), or None if invalid
    """
    if len(jn_start_list) != len(jn_end_list):
        return None

    exons = list(zip(jn_start_list, jn_end_list))

    cds_parts = []

    for exon_start, exon_end in exons:
        exon_start_int = int(exon_start)
        exon_end_int = int(exon_end)
        seq_start_int = int(seq_start)

        # Map 1-based genomic coordinates to 0-based region indices
        i_start = exon_start_int - seq_start_int
        i_end = exon_end_int - seq_start_int + 1

        if i_start < 0 or i_end > len(region):
            return None

        exon_seq = region[i_start:i_end]
        cds_parts.append(exon_seq)

    spliced = ''.join(cds_parts).upper()

    # RC if minus strand
    if strand == '-':
        spliced = str(reverse_complement(spliced))

    return spliced


def is_start_codon(codon):
    """Check if codon is a valid start codon."""
    return codon.upper() in ('ATG', 'TTG', 'CTG')


def is_stop_codon(codon):
    """Check if codon is a stop codon."""
    return codon.upper() in ('TAA', 'TAG', 'TGA')


def count_internal_stops(seq):
    """Count stop codons in frame (every 3rd codon, excluding last)."""
    if len(seq) < 6:
        return 0

    count = 0
    for i in range(0, len(seq) - 3, 3):
        codon = seq[i:i+3]
        if is_stop_codon(codon):
            count += 1
    return count


def get_downstream_triplet(region, seq_start, jn_start_list, jn_end_list, strand):
    """
    Retrieve the codon immediately downstream of the CDS in transcript sense.

    For plus strand: bases after the last exon end.
    For minus strand: bases before the first exon start (reverse-complemented).

    Returns the downstream codon on the sense strand, or None if out-of-bounds.
    """
    if not jn_start_list or not jn_end_list:
        return None

    seq_start_int = int(seq_start)

    if strand == '+':
        last_end = int(jn_end_list[-1])
        i_start = last_end - seq_start_int + 1
        i_end = i_start + 3
        if i_start < 0 or i_end > len(region):
            return None
        triplet = region[i_start:i_end]
        return triplet.upper()
    else:
        first_start = int(jn_start_list[0])
        i_start = first_start - seq_start_int - 3
        i_end = i_start + 3
        if i_start < 0 or i_end > len(region):
            return None
        triplet = region[i_start:i_end]
        return str(reverse_complement(triplet)).upper()


def validate_cds(cds_seq, require_terminal_stop=True):
    """
    Validate CDS sequence.

    Convention: CDS should include stop codon unless overridden.

    Args:
        cds_seq: CDS sequence to validate

    Returns:
        (is_valid, reason)
    """
    if not cds_seq or len(cds_seq) < 6:
        return False, "too_short"

    if len(cds_seq) % 3 != 0:
        return False, "not_divisible_by_3"

    if not is_start_codon(cds_seq[:3]):
        return False, "no_start_codon"

    # Check for internal stop codons (all but the last codon)
    num_internal_stops = count_internal_stops(cds_seq[:-3])
    if num_internal_stops > 0:
        return False, "internal_stops"

    # Check terminal stop codon (optional)
    terminal_codon = cds_seq[-3:]
    if is_stop_codon(terminal_codon):
        return True, "valid"

    if require_terminal_stop:
        return False, "no_terminal_stop"
    else:
        # Accept without terminal stop when allowed
        return True, "valid_without_terminal_stop"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_fasta', help='Reference genome FASTA')
    parser.add_argument('splice_table', help='Input splice_table file')
    parser.add_argument('--output_splice_table', required=True,
                        help='Output filtered splice_table')
    parser.add_argument('--output_sequence', required=True,
                        help='Output sequence file')
    parser.add_argument('--log', default=None,
                        help='Optional log file for statistics')
    parser.add_argument('--auto_missing_stop_threshold', type=float, default=0.9,
                        help='Auto mode: if proportion of invalid entries with no terminal stop ≥ threshold, extend coordinates by 3 where downstream stop exists (default: 0.9)')

    args = parser.parse_args()

    # Load reference genome
    print(f"Loading reference FASTA: {args.reference_fasta}", file=sys.stderr)
    sequences = load_fasta_index(args.reference_fasta)
    print(f"  Loaded {len(sequences)} sequences", file=sys.stderr)

    # Track statistics
    stats = defaultdict(int)
    reasons = defaultdict(int)

    # Cache entries for possible auto-fallback revalidation
    entries = []
    valid_lines_splice = []
    valid_lines_seq = []

    # Process splice_table
    print(f"Processing: {args.splice_table}", file=sys.stderr)
    with open(args.splice_table, 'r') as f_splice:
        for _, line_splice in enumerate(f_splice, 1):
            stats['total'] += 1

            line_splice = line_splice.rstrip('\n')
            cols = line_splice.split('\t')

            if len(cols) < 8:
                stats['invalid_format'] += 1
                reasons['invalid_format'] += 1
                continue

            chrom = cols[2]
            strand = cols[3]
            flank_start = cols[4]
            flank_end = cols[5]

            # Columns 6 and 7 contain comma-separated exon positions
            jn_end_str = cols[6]    # comma-separated exon ends (e.g., "11227,12811,13159,")
            jn_start_str = cols[7]  # comma-separated exon starts (e.g., "11100,12700,13050,")

            # Parse comma-separated lists, filter empty strings
            jn_end_list = [x for x in jn_end_str.split(',') if x]
            jn_start_list = [x for x in jn_start_str.split(',') if x]

            if not jn_start_list or not jn_end_list:
                stats['no_exons'] += 1
                reasons['no_exons'] += 1
                continue

            # Fetch flanked region
            region = get_flanked_region(chrom, flank_start, flank_end, strand, sequences)
            if region is None:
                stats['fetch_failed'] += 1
                reasons['fetch_failed'] += 1
                continue

            # Splice exons
            cds_seq = splice_exons(region, flank_start, jn_start_list, jn_end_list, strand)
            if cds_seq is None:
                stats['splice_failed'] += 1
                reasons['splice_failed'] += 1
                continue

            # Validate (initial pass always requires terminal stop)
            is_valid, reason = validate_cds(cds_seq, require_terminal_stop=True)

            downstream_triplet = None
            if reason == 'no_terminal_stop':
                # For diagnostics, check downstream triplet
                downstream_triplet = get_downstream_triplet(region, flank_start, jn_start_list, jn_end_list, strand)
                if downstream_triplet and is_stop_codon(downstream_triplet):
                    stats['downstream_stop_present'] += 1
                elif downstream_triplet:
                    stats['downstream_stop_absent'] += 1
                else:
                    stats['downstream_stop_oob'] += 1

            # Record entry for potential revalidation in auto mode
            entries.append({
                'line_splice': line_splice,
                'chrom': chrom,
                'strand': strand,
                'flank_start': flank_start,
                'flank_end': flank_end,
                'region': region,
                'cds_seq': cds_seq,
                    'jn_start_list': jn_start_list,
                    'jn_end_list': jn_end_list,
                'reason': reason,
                'initial_valid': is_valid,
            })

            if is_valid:
                stats['valid'] += 1
                reasons[reason] += 1

                # Build sequence line
                # Format: chrom:start-end TAB sequence (flanked region)
                seq_header = f"{chrom}:{flank_start}-{flank_end}"
                seq_line = f"{seq_header}\t{region}"

                valid_lines_splice.append(line_splice)
                valid_lines_seq.append(seq_line)
            else:
                stats['invalid'] += 1
                reasons[reason] += 1

    # Auto-extension: if most invalids are due to missing terminal stop, extend coordinates
    fallback_used = False
    if stats['invalid'] > 0:
        no_stop = reasons.get('no_terminal_stop', 0)
        proportion_no_stop = no_stop / stats['invalid']
        if proportion_no_stop >= args.auto_missing_stop_threshold:
            fallback_used = True

            # Reset counters and collections for second pass
            stats['valid'] = 0
            stats['invalid'] = 0
            valid_lines_splice = []
            valid_lines_seq = []
            reasons = defaultdict(int)

            for ent in entries:
                updated_line_splice = ent['line_splice']
                updated_cds = ent['cds_seq']
                did_extend = False

                if ent['reason'] == 'no_terminal_stop':
                    dt = get_downstream_triplet(ent['region'], ent['flank_start'], ent['jn_start_list'], ent['jn_end_list'], ent['strand'])
                    if dt and is_stop_codon(dt):
                        # Extend exon boundary by 3 and update splice_table coordinates
                        jn_start_list = ent['jn_start_list'][:]
                        jn_end_list = ent['jn_end_list'][:]
                        if ent['strand'] == '+':
                            jn_end_list[-1] = str(int(jn_end_list[-1]) + 3)
                        else:
                            jn_start_list[0] = str(int(jn_start_list[0]) - 3)

                        # Rebuild the splice_table line with updated exon lists
                        cols = updated_line_splice.split('\t')
                        if len(cols) >= 8:
                            cols[6] = ','.join(jn_end_list)
                            cols[7] = ','.join(jn_start_list)
                            updated_line_splice = '\t'.join(cols)
                            updated_cds = updated_cds + dt
                            did_extend = True

                # Validate after potential extension; require terminal stop in second pass
                if did_extend:
                    is_valid, reason = validate_cds(updated_cds, require_terminal_stop=True)
                else:
                    # Keep invalid if we couldn't extend to include a stop
                    is_valid, reason = (False, 'no_terminal_stop')

                if is_valid:
                    stats['valid'] += 1
                    reasons[reason] += 1
                    seq_header = f"{ent['chrom']}:{ent['flank_start']}-{ent['flank_end']}"
                    seq_line = f"{seq_header}\t{ent['region']}"
                    valid_lines_splice.append(updated_line_splice)
                    valid_lines_seq.append(seq_line)
                else:
                    stats['invalid'] += 1
                    reasons[reason] += 1

            # Always include entries that were already valid in the first pass
            for ent in entries:
                if ent['initial_valid']:
                    stats['valid'] += 1
                    reasons['valid'] += 1
                    seq_header = f"{ent['chrom']}:{ent['flank_start']}-{ent['flank_end']}"
                    seq_line = f"{seq_header}\t{ent['region']}"
                    valid_lines_splice.append(ent['line_splice'])
                    valid_lines_seq.append(seq_line)

    # Write outputs
    print("Writing outputs...", file=sys.stderr)
    with open(args.output_splice_table, 'w') as f:
        f.write('\n'.join(valid_lines_splice))
        if valid_lines_splice:
            f.write('\n')

    with open(args.output_sequence, 'w') as f:
        f.write('\n'.join(valid_lines_seq))
        if valid_lines_seq:
            f.write('\n')

    # Report statistics
    report = f"""
CDS Extraction & Validation Report
===================================
Total entries:     {stats['total']}
Valid:             {stats['valid']}
Invalid:           {stats['invalid']}

Failure reasons:
"""
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        if reason != 'valid':
            report += f"  {reason:30s}: {count:6d}\n"

    # Diagnostics: downstream stop presence when terminal stop missing
    if stats.get('downstream_stop_present') or stats.get('downstream_stop_absent') or stats.get('downstream_stop_oob'):
        report += "\nDownstream stop diagnostics (sense codon immediately after CDS):\n"
        report += f"  stop_present                 : {stats.get('downstream_stop_present', 0):6d}\n"
        report += f"  stop_absent                  : {stats.get('downstream_stop_absent', 0):6d}\n"
        report += f"  out_of_bounds                : {stats.get('downstream_stop_oob', 0):6d}\n"

    if fallback_used:
        report += "\nAuto-extension used: extended CDS coordinates to include terminal stops where present downstream.\n"

    report += f"""
Output files:
  Splice table: {args.output_splice_table} ({stats['valid']} entries)
  Sequence:     {args.output_sequence} ({stats['valid']} entries)
"""

    print(report, file=sys.stderr)

    if args.log:
        with open(args.log, 'w') as f:
            f.write(report)


if __name__ == '__main__':
    main()
