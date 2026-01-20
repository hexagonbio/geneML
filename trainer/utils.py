###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import re
from math import ceil

import numpy as np
from sklearn.metrics import average_precision_score

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP3 = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.
OUT_MAP5 = np.asarray([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])

OUT_MAP7 = np.asarray([[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0]])

OUT_MAPS = {3: OUT_MAP3, 5: OUT_MAP5, 7: OUT_MAP7}


def ceil_div(x, y):

    return int(ceil(float(x)/y))


# handle non-standard bases
# mapping based on first choice at http://www.hgmd.cf.ac.uk/docs/nuc_lett.html
FROMCHARS = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
TOCHARS = b'13230031002010000434001020'
TRANSLATION_TABLE = bytes.maketrans(FROMCHARS, TOCHARS)


def create_datapoints(seq, strand, tx_start, tx_end, jn_start, jn_end, seq_start, SL, CL_max, num_classes):
    # This function first converts the sequence into an integer array, where
    # A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    # negative, then reverse complementing is done. The splice junctions
    # are also converted into an array of integers, where 0, 1, 2, -1
    # correspond to no splicing, acceptor, donor and missing information
    # respectively. It then calls reformat_data and one_hot_encode
    # and returns X, Y which can be used by Keras models.

    # The provided sequence (gene + meaningful flanks) is used as-is.
    # Windows are extracted with CL_max/2 context on each side of SL-sized blocks.
    # When windows extend beyond the sequence, they are padded with Ns (encoded as 0)
    # to mark the boundary of relevant genomic context.

    seq = seq.upper().translate(TRANSLATION_TABLE)

    # Convert genomic coordinates to sequence-relative coordinates
    seq_start = int(seq_start)
    tx_start = int(tx_start)
    tx_end = int(tx_end)
    tx_start_rel = tx_start - seq_start  # Sequence-relative CDS start
    tx_end_rel = tx_end - seq_start      # Sequence-relative CDS end

    jn_start = list(map(lambda x: list(map(int, re.split(b',', x)[:-1])), jn_start))
    jn_end = list(map(lambda x: list(map(int, re.split(b',', x)[:-1])), jn_end))

    if strand == b'+':

        X0 = np.asarray(list(map(int, list(seq.decode()))))
        Y0 = [-np.ones(tx_end_rel-tx_start_rel+1) for t in range(1)]

        for t in range(1):

            if len(jn_start[t]) > 0:
                Y0[t] = np.zeros(tx_end_rel-tx_start_rel+1)

                if num_classes >= 6:
                    for exon_start, exon_end in zip(jn_end[t], jn_start[t]):
                        for c in range(exon_start, exon_end):
                            Y0[t][c-tx_start] = 5  # exonic
                    for intron_start, intron_end in zip(jn_start[t][:-1], jn_end[t][1:]):
                        for c in range(intron_start, intron_end):
                            Y0[t][c-tx_start] = 6  # intronic

                for c in jn_start[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][c-tx_start] = 2
                for c in jn_end[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][c-tx_start] = 1
                    # Ignoring junctions outside annotated tx start/end sites

                if num_classes >= 4:
                    Y0[t][jn_end[t][0]-tx_start] = 3  # start codon
                    Y0[t][jn_start[t][-1]-tx_start] = 4  # stop codon

    elif strand == b'-':

        X0 = (5-np.asarray(list(map(int, list(seq.decode()[::-1]))))) % 5  # Reverse complement
        Y0 = [-np.ones(tx_end_rel-tx_start_rel+1) for t in range(1)]

        for t in range(1):

            if len(jn_start[t]) > 0:
                Y0[t] = np.zeros(tx_end_rel-tx_start_rel+1)

                if num_classes >= 6:
                    for exon_start, exon_end in zip(jn_end[t], jn_start[t]):
                        for c in range(exon_start, exon_end):
                            Y0[t][tx_end-c] = 5  # exonic
                    for intron_start, intron_end in zip(jn_start[t][:-1], jn_end[t][1:]):
                        for c in range(intron_start, intron_end):
                            Y0[t][tx_end-c] = 6  # intronic

                for c in jn_end[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end-c] = 2
                for c in jn_start[t]:
                    if tx_start <= c <= tx_end:
                        Y0[t][tx_end-c] = 1

                if num_classes >= 4:
                    Y0[t][tx_end-jn_start[t][-1]] = 3  # start codon
                    Y0[t][tx_end-jn_end[t][0]] = 4  # stop codon

    else:
        assert False, 'failed: False'

    Xd, Yd = reformat_data(X0, Y0, SL, CL_max, tx_start_rel)
    X, Y = one_hot_encode(Xd, Yd, num_classes)

    return X, Y


def reformat_data(X0, Y0, SL, CL_max, tx_start):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are CL_max/2 on either side
    # of the SL nucleotides of interest.
    #
    # Sequence structure: [upstream_flank][CDS][downstream_flank]
    # X0 contains the full sequence. Y0 contains labels only for the CDS region.
    # tx_start is the sequence-relative position where the CDS starts.
    # Windows are extracted with stride SL over the CDS region, using flanks as context.
    # When a window extends beyond the sequence, it is padded with zeros (Ns).

    num_points = ceil_div(len(Y0[0]), SL)

    Xd = np.zeros((num_points, SL+CL_max))
    Yd = [-np.ones((num_points, SL)) for t in range(1)]

    # Pad Y0 to cover the final partial SL segment
    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1) for t in range(1)]

    half_ctx = CL_max // 2
    for i in range(num_points):
        # Y coordinates are CDS-relative (0-indexed within CDS)
        # X coordinates must be sequence-relative (0-indexed within full sequence)
        # For window i, we want to predict Y[SL*i : SL*(i+1)]
        # which corresponds to sequence positions [tx_start + SL*i, tx_start + SL*(i+1)]

        # Input window should span [tx_start + SL*i - half_ctx, tx_start + SL*(i+1) + half_ctx)
        left = tx_start + SL*i - half_ctx
        right = tx_start + SL*(i+1) + half_ctx

        pad_left = max(0, -left)
        pad_right = max(0, right - len(X0))

        s = max(0, left)
        e = min(len(X0), right)
        window = X0[s:e]
        if pad_left or pad_right:
            window = np.pad(window, [pad_left, pad_right], mode='constant', constant_values=0)
        Xd[i] = window

    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i:SL*(i+1)]

    return Xd, Yd


def one_hot_encode(Xd, Yd, num_classes):

    return IN_MAP[Xd.astype('int8')], \
           [OUT_MAPS[num_classes][Yd[t].astype('int8')] for t in range(1)]


def print_topl_statistics(y_true, y_pred, print_fn=print):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:

        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) / float(min(len(idx_pred), len(idx_true)))]
        threshold += [sorted_y_pred[-int(top_length*len(idx_true))]]

    auprc = average_precision_score(y_true, y_pred)

    print_fn(("%.4f\t\033[91m%.4f\t\033[0m%.4f\t%.4f\t\033[94m%.4f\t\033[0m"
              + "%.4f\t%.4f\t%.4f\t%.4f\t%d") % (
                 topkl_accuracy[0], topkl_accuracy[1], topkl_accuracy[2],
                 topkl_accuracy[3], auprc, threshold[0], threshold[1],
                 threshold[2], threshold[3], len(idx_true))
             )

    return topkl_accuracy[1]


def print_basic_statistics(y_true, y_pred, print_fn=print):
    # Prints the following information: auprc, number of true sites.

    idx_true = np.nonzero(y_true == 1)[0]
    auprc = average_precision_score(y_true, y_pred)

    print_fn(("%.4f\t%d") % (auprc, len(idx_true)))

    return auprc
