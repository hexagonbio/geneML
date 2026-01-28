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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    parser.add_argument('--max-total-genes', type=int, required=False,
                        help='Optional cap on total genes (after per-genome sampling and shuffling)')
    parser.add_argument('--outfile', type=str, required=False)
    parser.add_argument('--multigenome', required=False, action='store_true')
    parser.add_argument('--multigenome-glob', type=str, required=False, default='*')
    parser.add_argument('--multigenome-list', type=str, required=False)

    parser.add_argument('--max-genes-per-genome', type=int, required=False,
                        help='Cap the number of genes sampled from each genome before shuffling')

    parser.add_argument('--seed', type=int, required=False, default=42,
                        help='Deterministic seed for sampling/shuffling (default: 42)')

    parser.add_argument('--num-classes', type=int, choices=[3, 5, 7], required=False, default=7)

    parser.add_argument('--num-threads', type=int, required=False, default=8,
                        help='Number of threads for parallel processing (default: 8)')

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
datafiles = sorted(datafiles)
print('num datafiles:', len(datafiles))

# Build a global shuffled list of (datafile, gene_idx) pairs across genomes
CHUNK_SIZE = args.chunk_size
all_genes = []
rng = random.Random(args.seed)

for datafile in datafiles:
    with h5py.File(datafile, 'r') as h5f:
        total_genes = h5f['SEQ'].shape[0]

    max_from_this = args.max_genes_per_genome if args.max_genes_per_genome else total_genes
    max_from_this = min(max_from_this, total_genes)

    chosen_indices = rng.sample(range(total_genes), max_from_this) if max_from_this < total_genes else list(range(total_genes))
    for idx in chosen_indices:
        all_genes.append((datafile, idx))

    print(datafile, 'total genes:', total_genes, 'selected:', len(chosen_indices))

rng.shuffle(all_genes)

if args.max_total_genes:
    all_genes = all_genes[:args.max_total_genes]

print('total selected genes (all genomes):', len(all_genes))

if args.outfile:
    outfile = args.outfile
else:
    outfile = ('dataset' + '_' + args.mode + '_' + str(args.CL_max) + '_' + str(args.SL) + '_' + args.suffix + '.h5')
h5f2 = h5py.File(os.path.join(args.data_dir, outfile), 'w')

def process_chunk(chunk_idx, chunk_genes, SL, CL_max):
    """Process a chunk of genes in parallel."""
    # Each thread needs its own H5 file handles
    h5_cache = {}

    def get_handle(path):
        if path not in h5_cache:
            h5_cache[path] = h5py.File(path, 'r')
        return h5_cache[path]

    X_batch = []
    Y_batch = [[] for t in range(1)]

    for datafile, gene_idx in chunk_genes:
        h5f = get_handle(datafile)
        X, Y = create_datapoints(
            h5f['SEQ'][gene_idx], h5f['STRAND'][gene_idx],
            h5f['CDS_START'][gene_idx], h5f['CDS_END'][gene_idx],
            h5f['JN_START'][gene_idx], h5f['JN_END'][gene_idx],
            h5f['SEQ_START'][gene_idx],
            SL=SL, CL_max=CL_max,
        )

        X_batch.extend(X)
        for t in range(1):
            Y_batch[t].extend(Y[t])

    # Close handles for this thread
    for h in h5_cache.values():
        h.close()

    # Convert to numpy arrays
    X_batch = np.asarray(X_batch).astype('int8')
    for t in range(1):
        Y_batch[t] = np.asarray(Y_batch[t]).astype('float32')

    return chunk_idx, X_batch, Y_batch

h5f2_offset = 0
num_chunks_total = int(np.ceil(len(all_genes) / float(CHUNK_SIZE))) if len(all_genes) else 0

print(f'Processing {num_chunks_total} chunks with {args.num_threads} processes...')

# Process chunks in batches to limit memory usage
batch_size = args.num_threads * 2  # Process 2x num_threads at a time
with ProcessPoolExecutor(max_workers=args.num_threads) as executor:
    for batch_start in range(0, num_chunks_total, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks_total)

        # Submit batch of jobs
        futures = {}
        for chunk_idx in range(batch_start, batch_end):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(all_genes))
            chunk_genes = all_genes[start:end]

            future = executor.submit(process_chunk, chunk_idx, chunk_genes, args.SL, args.CL_max)
            futures[future] = chunk_idx

        # Collect results for this batch
        results = {}
        for future in as_completed(futures):
            chunk_idx, X_batch, Y_batch = future.result()
            results[chunk_idx] = (X_batch, Y_batch)

        # Write results in order
        for chunk_idx in range(batch_start, batch_end):
            X_batch, Y_batch = results[chunk_idx]
            h5f2.create_dataset('X' + str(h5f2_offset), data=X_batch)
            h5f2.create_dataset('Y' + str(h5f2_offset), data=Y_batch)
            h5f2_offset += 1

        if batch_end % 100 == 0 or batch_end == num_chunks_total:
            print(f'Completed {batch_end}/{num_chunks_total} chunks')

h5f2.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
