# geneML
geneML is a deep learning–based tool for fungal gene prediction.

## Installation
The only requirement is python v3.9 or higher.

### Using virtualenv
Start with a fresh python virtual environment:
```bash
python -m venv geneml
. geneml/bin/activate
# Now install the latest release from PyPI:
pip install geneml
```

### Using conda
Or use a conda environment:
```bash
conda create -n geneml -c conda-forge python=3.13 pip
conda activate geneml
# Now install the latest release from PyPI:
pip install geneml
```

### Directly from the repo
Or install directly from this repo (to get access to the latest changes):
```bash
git clone https://github.com/hexagonbio/geneML.git
pip install geneML
```
## Frequent options
Basic command:
```bash
geneml genome.fasta
```
To enable verbose mode:
```bash
geneml genome.fasta -v
```
To change the output path:
```bash
geneml genome.fasta -o genome_output.gff3
```
To run only on selected contigs:
```bash
geneml genome.fasta --contigs-filter NC_092406.1,NC_092407.1
```
To write nucleotide and protein sequences of the predicted genes (one sequence per transcript):
```bash
geneml genome.fasta -g genes.fna
geneml genome.fasta -p proteins.faa
```
By default, geneML outputs multiple transcripts per locus (if there are multiple high scoring options).<br>
You can change the maximum number of transcripts produced, for example forcing to output only the best transcript:
```bash
geneml genome.fasta --max-transcripts 1
```
With enough input data, GeneML dynamically determines the minimum score threshold for reporting genes and transcripts.<br>
You can override this threshold manually, for example:
```bash
geneml genome.fasta --min-gene-score 0.5
```

## Output
geneML writes gene annotations in GFF3 format.
### Fields
For each predicted gene, transcript, exon and CDS feature, the GFF3 includes:<br>
```
contig_name  source  feature_type  start  end  feature_score  strand  phase  identifiers
```
Note: As geneML does not include untranslated regions in its predictions, CDS features are identical to exon features (except for the added phase attribute).<br>
For more information on the GFF3 format, see: https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
### Identifiers
Each feature has a unique `ID` and, for child features, a `Parent` attribute specifying the parent ID.<br>
| Feature    | Example ID              |
|------------|-------------------------|
| Gene       | GML000001               |
| Transcript | GML000001_mRNA1         |
| Exon       | GML000001_mRNA1_exon1   |
| CDS        | GML000001_mRNA1_CDS1    |

### Additional information
The directive ##geneml-mean-gene-score (found at the top of the GFF file) stores the average score of predicted genes.

## Gene scoring and selection
geneML initially predicts many potential gene structures for the same locus.<br>
Every such gene call is assigned a quality score, ranging from 0 (very low confidence) to 1 (highest confidence).<br>
The quality score measures how well a gene call agrees with the per-position class probabilities that are output by geneML's convolutional neural network.<br>

The quality score is calculated as:<br>
Score = (A + B) / 2, where:

- A. Event consistency score: How well exon/intron regions match the model's is_exon/is_intron class probabilites
- B. Border score: Average model confidence at CDS start, CDS end and splice site positions

geneML applies the following strategy for select the best gene call(s):<br>
For each group of overlapping predictions, we filter out gene calls that do not score within a margin of 0.2 from the maximum score in the group.<br>
We build an initial selection that contains a seed gene call, which is the highest-scoring
gene call at the earliest start position, and any gene calls that are valid alternatives to it.<br>
A gene call is considered a valid alternative to another if:<br>
- The combined probabilities of any added events (gene start, gene end and splice sites) exceed those of any removed events by at least 0.2, or:
- There is a single event difference with a probability of at least 0.2.<br>
From this selection, we define the gene call with the longest CDS as the primary transcript. We define the other gene calls as alternative
transcripts, and classify the type of alternative transcript by comparing their gene structure to the
primary transcript.<br>

We remove low-confidence transcripts based on a minimum score.<br>
If enough input sequence is provided (at least 100 kb) and at least 100 loci are predicted, this minimum score is set dynamically.<br>
For a given genome, quality scores typically follow a bimodal distribution, with the first peak
representing short, low-confidence transcripts and the second peak representing longer,
high-confidence transcripts.<br>
The score cutoff is defined based on the local minimum between the two peaks.<br>
In absence of sufficient data or when no local minimum can be found, the default score threshold is 0.5.<br>
The threshold can be changed by setting --min-gene-score.

Scores included in the GFF3 output:
- Exon / CDS: Quality score for the exon
- Transcript: Quality score for the full gene call
- Gene: Highest transcript score for the locus

## Full Usage
```
geneml --help
usage: geneml [-h] [--version] [-o OUTPUT] [-g GENES] [-p PROTEINS] [--gene-id-prefix GENE_ID_PREFIX] [-m MODEL] [-cl CONTEXT_LENGTH] [-c CORES] [-v] [-d]
              [--cpu-only] [--strand {forward,reverse,both}] [--contigs-filter CONTIGS_FILTER] [--write-raw-scores]
              [--max-transcripts MAX_TRANSCRIPTS] [--allow-opposite-strand-overlaps {true,false}] [--min-gene-score MIN_GENE_SCORE]
              [--min-exon-size MIN_EXON_SIZE] [--max-exon-size MAX_EXON_SIZE] [--min-intron-size MIN_INTRON_SIZE]
              [--max-intron-size MAX_INTRON_SIZE] [--cds-start-min-score CDS_START_MIN_SCORE]
              [--cds-end-min-score CDS_END_MIN_SCORE] [--exon-start-min-score EXON_START_MIN_SCORE]
              [--exon-end-min-score EXON_END_MIN_SCORE] [--gene-candidates GENE_CANDIDATES]
              sequence

geneML 1.0.0

positional arguments:
  sequence              Sequence file in FASTA/GenBank/EMBL format.

options:
  -h, --help            Show this help message and exit.
  --version             Show version number and exit.
  -o OUTPUT, --output OUTPUT
                        Gene annotations output path (default: based on input filename).
  -g GENES, --genes GENES
                        Gene sequences output path (default: None).
  -p PROTEINS, --proteins PROTEINS
                        Protein sequences output path (default: None).
  --gene-id-prefix GENE_ID_PREFIX
                        Prefix for gene IDs in output (default: None).
  -m MODEL, --model MODEL
                        Path to model file (default: models/geneML_default.keras).
  -cl CONTEXT_LENGTH, --context-length CONTEXT_LENGTH
                        Context length of the model.
  -c CORES, --cores CORES
                        Number of cores to use for processing (default: all available).

advanced options:
  -v, --verbose         Enable verbose mode.
  -d, --debug           Enable debug mode.
  --cpu-only            Use CPU only for inference, disable GPU usage.
  --strand {forward,reverse,both}
                        On which strand to predict genes (default: both).
  --contigs-filter CONTIGS_FILTER
                        Run only on selected contigs (comma separated string).
  --write-raw-scores    Instead of running gene calling, output the raw model scores as a .seg file.
  --max-transcripts MAX_TRANSCRIPTS
                        Maximum number of transcripts per gene (default: 5).
  --allow-opposite-strand-overlaps {true,false}
                        Predict overlapping genes on opposite strands (default: true).
  --min-gene-score MIN_GENE_SCORE
                        Minimum gene score for gene reporting. Can be a float value or 'dynamic' (default: dynamic). Dynamic mode
                        requires >=100,000 bp total input.
  --min-exon-size MIN_EXON_SIZE
                        Minimum exon size (default: 1).
  --max-exon-size MAX_EXON_SIZE
                        Maximum exon size (default: 30000).
  --min-intron-size MIN_INTRON_SIZE
                        Minimum intron size (default: 10).
  --max-intron-size MAX_INTRON_SIZE
                        Maximum intron size (default: 400).
  --cds-start-min-score CDS_START_MIN_SCORE
                        Minimum model score for considering a CDS start (default: 0.01).
  --cds-end-min-score CDS_END_MIN_SCORE
                        Minimum model score for considering a CDS end (default: 0.01).
  --exon-start-min-score EXON_START_MIN_SCORE
                        Minimum model score for considering an exon start (default: 0.01).
  --exon-end-min-score EXON_END_MIN_SCORE
                        Minimum model score for considering an exon end (default: 0.01).
  --gene-candidates GENE_CANDIDATES
                        Maximum number of gene candidates to consider per locus (default: 5000).
```
