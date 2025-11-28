#!/bin/bash

echo processing $splice_table $ref_genome

# First nucleotide not included by BEDtools
cat $splice_table | awk '{print $3"\t"($5-1)"\t"$6}' > $splice_table.bed

bedtools getfasta -bed $splice_table.bed -fi $ref_genome -fo $sequence -tab

rm $splice_table.bed
