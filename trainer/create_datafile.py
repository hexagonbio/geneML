###############################################################################
# This parser takes as input the text files canonical_dataset.txt and
# canonical_sequence.txt, and produces a .h5 file datafile_{}_{}.h5,
# which will be later processed to create dataset_{}_{}.h5. The file
# dataset_{}_{}.h5 will have datapoints of the form (X,Y), and can be
# understood by Keras models.
###############################################################################

import os
import re
import sys
import time

import h5py
import numpy as np

start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all'], "failed: sys.argv[1] in ['train', 'test', 'all']"
# assert sys.argv[2] in ['0', '1', 'all']

mode = sys.argv[1]
suffix = sys.argv[2]
data_dir = sys.argv[3]
sequence = sys.argv[4]
splice_table = sys.argv[5]

###############################################################################

NAME = []      # Gene symbol
PARALOG = []   # 0 if no paralogs exist, 1 otherwise
CHROM = []     # Chromosome number
STRAND = []    # Strand in which the gene lies (+ or -)
CDS_START = []  # Position where cds starts
CDS_END = []    # Position where cds ends
JN_START = []  # Positions where canonical exons end
JN_END = []    # Positions where canonical exons start
SEQ = []       # Nucleotide sequence

fpr2 = open(sequence, 'r')

count = 0
with open(splice_table, 'r') as fpr1:
    for i, line1 in enumerate(fpr1):

        line2 = fpr2.readline()

        data1 = re.split('\n|\t', line1)[:-1]
        data2 = re.split('\n|\t|:|-', line2)[:-1]

        assert data1[2] == data2[0], 'failed: data1[2] == data2[0]'

        # count += 1
        # if count>2000: break
        # if not data1[2].endswith(('0', '2', '4', '6', '8')): continue
        if (mode == 'train' and i % 5 == 0) or \
           (mode == 'test' and i % 5 != 0):
            continue

        # if (sys.argv[2] != data1[1]) and (sys.argv[2] != 'all'):
        #     continue

        NAME.append(data1[0])
        PARALOG.append(int(data1[1]))
        CHROM.append(data1[2])
        STRAND.append(data1[3])
        CDS_START.append(data1[4])
        CDS_END.append(data1[5])
        JN_START.append(data1[6::2])
        JN_END.append(data1[7::2])
        SEQ.append(data2[3])

fpr1.close()
fpr2.close()

###############################################################################

h5f = h5py.File(os.path.join(data_dir, 'datafile'
                + '_' + mode + '_' + suffix
                + '.h5'), 'w')


def asarray(arr):
    return np.asarray(arr, dtype='S')


h5f.create_dataset('NAME', data=asarray(NAME))
h5f.create_dataset('PARALOG', data=asarray(PARALOG))
h5f.create_dataset('CHROM', data=asarray(CHROM))
h5f.create_dataset('STRAND', data=asarray(STRAND))
h5f.create_dataset('CDS_START', data=asarray(CDS_START))
h5f.create_dataset('CDS_END', data=asarray(CDS_END))
h5f.create_dataset('JN_START', data=asarray(JN_START))
h5f.create_dataset('JN_END', data=asarray(JN_END))
h5f.create_dataset('SEQ', data=asarray(SEQ))

h5f.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################

