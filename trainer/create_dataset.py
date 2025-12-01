###############################################################################
'''This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models.'''
###############################################################################

import argparse
import glob
import os
import random
import time

import h5py
import numpy as np
from utils import create_datapoints

# from constants import *

start_time = time.time()

# assert sys.argv[1] in ['train', 'test', 'all']
# assert sys.argv[2] in ['0', '1', 'all']


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--CL_max', type=int, required=True)
    parser.add_argument('--SL', type=int, required=True)

    parser.add_argument('--chunk-size', type=int, required=False, default=100)
    parser.add_argument('--max-num-chunks', type=int, required=False)
    parser.add_argument('--outfile', type=str, required=False)
    parser.add_argument('--multigenome', required=False, action='store_true')
    parser.add_argument('--multigenome-glob', type=str, required=False, default='*')
    parser.add_argument('--multigenome-list', type=str, required=False)

    parser.add_argument('--num-classes', type=int, choices=[3, 5, 7], required=False, default=7)

    args, _ = parser.parse_known_args()
    return args


args = get_args()

if args.multigenome:
    if args.multigenome_list:
        paths = []
        genome_list = [line.strip() for line in open(args.multigenome_list)]
        datafiles = []
        for g in genome_list:
            datafiles.extend(glob.glob(os.path.join(args.data_dir, 'datafile' + '_' + args.mode + f'_{args.multigenome_glob}{g}.h5')))
    else:
        datafiles = sorted(glob.glob(os.path.join(args.data_dir, 'datafile' + '_' + args.mode + f'_{args.multigenome_glob}.h5')))
else:
    datafiles = [os.path.join(args.data_dir, 'datafile' + '_' + args.mode + '_' + str(args.CL_max) + '_' + args.suffix + '.h5')]
print('num datafiles:', len(datafiles))

if args.outfile:
    outfile = args.outfile
else:
    outfile = ('dataset' + '_' + args.mode + '_' + str(args.CL_max) + '_' + str(args.SL) + '_' + args.suffix + '.h5')
h5f2 = h5py.File(os.path.join(args.data_dir, outfile), 'w')

CHUNK_SIZE = args.chunk_size
h5f2_offset = 0
for datafile in datafiles:

    h5f = h5py.File(datafile, 'r')
    SEQ = h5f['SEQ'][:]
    STRAND = h5f['STRAND'][:]
    CDS_START = h5f['CDS_START'][:]
    CDS_END = h5f['CDS_END'][:]
    JN_START = h5f['JN_START'][:]
    JN_END = h5f['JN_END'][:]
    SEQ_START = h5f['SEQ_START'][:]
    h5f.close()

    total_chunks = SEQ.shape[0]//CHUNK_SIZE
    if args.max_num_chunks:
        num_chunks = min(total_chunks, args.max_num_chunks)
        chunk_choices = random.sample(list(range(total_chunks)), num_chunks)
    else:
        num_chunks = total_chunks
        chunk_choices = list(range(num_chunks))

    print(datafile, 'total num_chunks:', total_chunks, 'num_chunks:', num_chunks)

    for offset, i in enumerate(chunk_choices):
        # Each dataset has CHUNK_SIZE genes

        if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
            NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0] % CHUNK_SIZE
        else:
            NEW_CHUNK_SIZE = CHUNK_SIZE

        X_batch = []
        Y_batch = [[] for t in range(1)]

        for j in range(NEW_CHUNK_SIZE):

            idx = i*CHUNK_SIZE + j

            X, Y = create_datapoints(
                SEQ[idx], STRAND[idx],
                CDS_START[idx], CDS_END[idx],
                JN_START[idx], JN_END[idx],
                SEQ_START[idx],
                SL=args.SL, CL_max=args.CL_max, num_classes=args.num_classes,
            )

            X_batch.extend(X)
            for t in range(1):
                Y_batch[t].extend(Y[t])

        X_batch = np.asarray(X_batch).astype('int8')
        for t in range(1):
            Y_batch[t] = np.asarray(Y_batch[t]).astype('float32')

        # h5f2.create_dataset('X' + str(i), data=X_batch)
        # h5f2.create_dataset('Y' + str(i), data=Y_batch)
        h5f2.create_dataset('X' + str(h5f2_offset+offset), data=X_batch)
        h5f2.create_dataset('Y' + str(h5f2_offset+offset), data=Y_batch)

    h5f2_offset += num_chunks

h5f2.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
