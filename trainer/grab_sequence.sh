#!/bin/bash

echo processing $splice_table $ref_genome

CLr=$((CL_max/2))
CLl=$(($CLr+1))
# First nucleotide not included by BEDtools

cat $splice_table | awk -v CLl=$CLl -v CLr=$CLr '{print $3"\t"($5-CLl)"\t"($6+CLr)}' > $splice_table.bed

bedtools getfasta -bed $splice_table.bed -fi $ref_genome -fo $sequence -tab

rm $splice_table.bed
