#!/bin/bash

set -ex

G=$1
DATA_DIR=$(realpath ${2:=./})
TSV_DIR=$(realpath ${3:=./})
FASTA_DIR=$(realpath ${4:=./})
echo DATA_DIR: $DATA_DIR
mkdir -p $DATA_DIR

if [ -f "$DATA_DIR/datafile_train_${G}.h5" ]; then
    exit
fi

genome_path=$FASTA_DIR/*$G*.f*a
samtools faidx $genome_path

export splice_table=$TSV_DIR/$G.tsv
export ref_genome=$genome_path
export data_dir=$DATA_DIR
export sequence=$DATA_DIR/${G}_canonical_sequence.txt
bash trainer/grab_sequence.sh

python trainer/create_datafile.py train $G $data_dir $sequence $splice_table
python trainer/create_datafile.py test $G $data_dir $sequence $splice_table

