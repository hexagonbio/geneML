#!/bin/bash

set -ex

G=$1
DATA_DIR=$(realpath ${2:=./})
TSV_DIR=$(realpath ${3:=./})
FASTA_DIR=$(realpath ${4:=./})
echo DATA_DIR: $DATA_DIR
mkdir -p $DATA_DIR

if [ -f "$DATA_DIR/datafile_all_${G}.h5" ]; then
    exit
fi

genome_path=$FASTA_DIR/*$G*.f*a
splice_table=$TSV_DIR/$G.tsv
new_splice_table=$TSV_DIR/new_$G.tsv
sequence=$DATA_DIR/${G}_canonical_sequence.txt

python trainer/extract_and_validate_cds.py $genome_path $splice_table \
    --output_splice_table $new_splice_table --output_sequence $sequence
python trainer/create_datafile.py all $G $DATA_DIR $sequence $new_splice_table
