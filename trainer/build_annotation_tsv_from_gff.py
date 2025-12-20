import gzip
import os
import sys

genome_id = sys.argv[1]
path = sys.argv[2]
data_dir = sys.argv[3]

if len(sys.argv) > 4:
    allowed_genes = set()
    with open(sys.argv[4]) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            allowed_genes.add(cols[0].replace('"', ''))
else:
    allowed_genes = None

data = {}
inferred_contig_sizes = {}
breakpoints = {}

with open(path) if not path.endswith('.gz') else gzip.open(path, mode='rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.strip().split('\t')
        chrom = cols[0]
        feature_type = cols[2]
        if feature_type != 'CDS':
            continue

        start = int(cols[3])
        end = int(cols[4])

        if end > inferred_contig_sizes.get(chrom, 0):
            inferred_contig_sizes[chrom] = end
        strand = cols[6]
        info = {}
        split_char = '=' if 'Parent' in cols[8] else ' '
        skip_line = False
        for row in cols[8].strip(';').split(';'):
            try:
                k, v = row.strip().split(split_char, 1)
                info[k] = v
            except ValueError:
                print(f'problems parsing {row} in {path}, line: {line.strip()}')
                skip_line = True
                break
        if skip_line:
            continue

        # Skip pseudogenes
        if info.get('pseudo') == 'true' or 'pseudogene' in info:
            continue

        if 'ID' in info:
            # ID=cds-XP_024343564.1 in GCF_002117355.1 gff
            if '-' in info['ID']:
                id = info['ID'].split('-')[1]
            else:
                id = info['ID']
        elif 'name' in info:
            id = info['name']
        elif 'Parent' in info:
            id = info['Parent']  # ncbi GCA_ genomes
        elif 'gene_id' in info:
            id = info['gene_id'].replace('"', '')  # stringtie format
        else:
            assert False, 'failed: False'
        key = chrom, strand, id
        if key not in data:
            data[key] = []
        data[key].append(tuple([start, end]))

        key = chrom, strand
        if key not in breakpoints:
            breakpoints[key] = []
        breakpoints[key].extend([start, end])

for v in breakpoints.values():
    v.sort()

with open(os.path.join(data_dir, f'{genome_id}.tsv'), 'w') as f:
    for (chrom, strand, name), exons in sorted(data.items()):

        if allowed_genes is not None and name not in allowed_genes:
            continue

        # ncbi can be in reverse order for reverse strand genes
        exons.sort()

        exon_ends = [x[1] for x in exons]  # already one-based
        exon_starts = [x[0] for x in exons]  # since parsed from gff, already one-based

        key = chrom, strand
        start = exon_starts[0]
        end = exon_ends[-1]

        # define utr to be 1000 bp outside of gene, unless there's a neighboring gene
        idx = breakpoints[key].index(start)
        start = max(start-1000, breakpoints[key][idx-1] if idx > 0 else 0)
        idx = breakpoints[key].index(end)
        end = min(end+1000, breakpoints[key][idx+1] if idx < len(breakpoints[key])-2 else 1e10)

        # skip genes at the ends of a contig
        if start < 2000:
            continue
        if end > inferred_contig_sizes[chrom]-2000:
            continue

        print(name.replace('"', ''),  # gene name
              0,  # paralogous, not used in downstream code
              chrom,
              strand,
              start, end,
              ''.join(f'{x},' for x in exon_ends),
               ''.join(f'{x},' for x in exon_starts),
              sep='\t', file=f)
