#!/bin/bash

set -ex

G=$1
CL=$2
DATA_DIR=$(realpath ${3:=./})
TSV_DIR=$(realpath ${4:=./})
FASTA_DIR=$(realpath ${5:=./})
echo DATA_DIR: $DATA_DIR
mkdir -p $DATA_DIR

if [ -f "$DATA_DIR/datafile_train_${CL}_${G}.h5" ]; then
    exit
fi

genome_path=$FASTA_DIR/*$G*.f*a
samtools faidx $genome_path

export splice_table=$TSV_DIR/$G.tsv
export CL_max=$CL
export ref_genome=$genome_path
export data_dir=$DATA_DIR
export sequence=$DATA_DIR/${G}_canonical_sequence.txt
bash trainer/grab_sequence.sh

python trainer/create_datafile.py train $CL_max $G $data_dir $sequence $splice_table
python trainer/create_datafile.py test $CL_max $G $data_dir $sequence $splice_table

