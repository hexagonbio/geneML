###############################################################################
# This file contains the code to train the GeneML model.
###############################################################################

import argparse
import os
import re
import sys
import time

# from lib.genes.gene_ml import loss_functions


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--context-length',
            type=int,
            required=True,
            )
    parser.add_argument(
            '--dataset-name',
            type=str,
            required=True,
            )
    parser.add_argument(
            '--train-path',
            type=str,
            required=True,
            )
    parser.add_argument(
        '--model-type',
        type=str, choices=['spliceai', 'gene_ml'],
        required=True,
    )
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
            '--job-name',
            type=str,
            required=True,
            )
    parser.add_argument(
        '--keras-save-path',
        type=str,
        default='./models/',
        help='Directory to save keras model files (default: ./models/)')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='number of times to go through the data, default=10')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size per GPU. If not specified, will auto-tune based on available VRAM.')
#    parser.add_argument(
#        '--learning-rate',
#        default=.01,
#        type=float,
#        help='learning rate for gradient descent, default=.01')
#    parser.add_argument(
#        '--verbosity',
#        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
#        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def main():
    os.environ["TF_USE_LEGACY_KERAS"] = "0"  # use keras v3 even though we have tf_keras installed

    import h5py
    import keras
    import numpy as np
    import tensorflow

    # works locally and on Vertex AI
    from model import GeneML
    from utils import clip_datapoints, print_basic_statistics, print_topl_statistics

    args = get_args()

    os.makedirs('Models', exist_ok=True)

    logname = ('GeneML' + str(args.context_length) + '_c' + args.dataset_name + '.log')
    local_logpath = './Models/' + logname

    def remove_ansi_codes(text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def tee(*args):
        print(*args)
        print(*[remove_ansi_codes(x) if type(x) is str else x for x in args], file=tee.f)

    tee.f = open(local_logpath, 'w')

    assert args.train_path, 'failed: args.train_path'
    CL_max = int(os.path.basename(args.train_path).split('_')[2])
    SL = int(os.path.basename(args.train_path).split('_')[3])

    ###############################################################################
    # Model
    ###############################################################################

    L = 32
    N_GPUS = args.num_gpus

    W = None
    AR = None
    DEFAULT_BATCH_SIZE = None  # Default batch size for auto-tuning starting point
    if int(args.context_length) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        DEFAULT_BATCH_SIZE = 18*N_GPUS
    elif int(args.context_length) == 240:
        W = np.asarray([11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4])
        DEFAULT_BATCH_SIZE = 18*N_GPUS
    elif int(args.context_length) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        DEFAULT_BATCH_SIZE = 18*N_GPUS
    elif int(args.context_length) == 768:
        W =  np.asarray([5,  5,  7,  11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1,  1,  1,  2,  2,  4,  4,  8,  16, 1])
        DEFAULT_BATCH_SIZE = 18*N_GPUS
    elif int(args.context_length) == 800:
        W =  np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1,  1,  1,  1,  4,  4,  4,  4,  10, 10])
        DEFAULT_BATCH_SIZE = 18*N_GPUS
    elif int(args.context_length) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10])
        DEFAULT_BATCH_SIZE = 12*N_GPUS
    elif int(args.context_length) == 5360:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 29, 29, 29, 29])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10, 15, 15, 15, 15])
        DEFAULT_BATCH_SIZE = 8*N_GPUS
    elif int(args.context_length) == 10080:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10, 25, 25, 25, 25])
        DEFAULT_BATCH_SIZE = 6*N_GPUS
    else:
        assert False, 'failed: int(args.context_length) does not match any known configuration'

    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit

    CL = 2 * np.sum(AR*(W-1))
    tee(CL, CL_max)
    assert CL <= CL_max and CL == int(args.context_length), 'failed: CL <= CL_max and CL == int(args.context_length)'
    tee("\033[1mContext nucleotides: %d\033[0m" % (CL))
    tee("\033[1mSequence length (output): %d\033[0m" % (SL))

    # Set batch size: use user-specified value or will auto-tune later
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size * N_GPUS
        tee(f"\033[1mUsing user-specified batch size: {BATCH_SIZE} ({args.batch_size} per GPU)\033[0m")
    else:
        BATCH_SIZE = None  # Will be set by auto-tuning

    if args.model_type == 'spliceai':
        num_classes = 3
        # loss = loss_functions.categorical_crossentropy_2d
        assert False, 'failed: False'
    elif args.model_type == 'gene_ml':
        @keras.saving.register_keras_serializable()
        def categorical_crossentropy_2d_gene_ml(y_true, y_pred):
            # Standard categorical cross entropy for sequence outputs
            kb = keras.ops
            return - kb.mean(y_true[:, :, 0] * kb.log(y_pred[:, :, 0] + 1e-10)
                             + y_true[:, :, 1] * kb.log(y_pred[:, :, 1] + 1e-10)
                             + y_true[:, :, 2] * kb.log(y_pred[:, :, 2] + 1e-10)
                             + y_true[:, :, 3] * kb.log(y_pred[:, :, 3] + 1e-10)
                             + y_true[:, :, 4] * kb.log(y_pred[:, :, 4] + 1e-10)
                             + y_true[:, :, 5] * kb.log(y_pred[:, :, 5] + 1e-10)
                             + y_true[:, :, 6] * kb.log(y_pred[:, :, 6] + 1e-10))

        num_classes = 7
        loss = categorical_crossentropy_2d_gene_ml
    else:
        assert False, 'failed: False'

    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    if N_GPUS > 1:
        # https://keras.io/guides/distributed_training/
        strategy = tensorflow.distribute.MirroredStrategy()
        tee('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = GeneML(L, W, AR, num_classes)
            model.compile(loss=loss, optimizer='adam')
    else:
        model = GeneML(L, W, AR, num_classes)
        model.compile(loss=loss, optimizer='adam')
    # model.summary()

    ###############################################################################
    # Training and validation
    ###############################################################################

    h5f = h5py.File(args.train_path, 'r')

    num_idx = len(h5f.keys())//2
    tee('num_idx:', num_idx)
    idx_all = np.random.permutation(num_idx)
    idx_train = idx_all[:int(0.9*num_idx)]
    idx_valid = idx_all[int(0.9*num_idx):]

    # Auto-tune batch size if not specified
    if args.batch_size is None:
        tee("\n\033[1mAuto-tuning batch size...\033[0m")

        def find_max_batch_size(start_batch_size):
            """Binary search for maximum batch size that fits in VRAM"""
            # Load a sample to test with
            test_idx = idx_train[0]
            X_test = h5f['X' + str(test_idx)][:]
            Y_test = h5f['Y' + str(test_idx)][:]
            Xc_test, Yc_test = clip_datapoints(X_test, Y_test, CL, N_GPUS, CL_max)

            batch_size = start_batch_size
            max_working = start_batch_size

            # First, find upper bound by doubling
            tee(f"Finding upper bound starting from {batch_size}...")
            while True:
                try:
                    tensorflow.keras.backend.clear_session()

                    # Test if this batch size fits
                    num_samples = min(batch_size, Xc_test.shape[0])
                    if num_samples < batch_size:
                        tee(f"  Sample has only {num_samples} datapoints, using that as max")
                        max_working = num_samples
                        break

                    tee(f"  Testing batch_size={batch_size}...", end=" ")
                    model.fit(Xc_test[:num_samples], Yc_test[:num_samples],
                             batch_size=batch_size, epochs=1, verbose=0)

                    tee("✓ fits")
                    max_working = batch_size
                    batch_size = min(batch_size * 2, Xc_test.shape[0])  # Double it

                    if batch_size == max_working:  # Can't go higher
                        break

                except (tensorflow.errors.ResourceExhaustedError,
                        tensorflow.errors.InternalError):
                    tee("✗ OOM")
                    # Now binary search between max_working and batch_size
                    break

            # Binary search to find the exact maximum
            low = max_working
            high = batch_size

            if high > low:
                tee(f"Binary searching between {low} and {high}...")
                while high - low > N_GPUS * 4:  # Stop when difference is small
                    mid = (low + high) // 2
                    # Round to multiple of N_GPUS for clean distribution
                    mid = (mid // N_GPUS) * N_GPUS

                    try:
                        tensorflow.keras.backend.clear_session()
                        tee(f"  Testing batch_size={mid}...", end=" ")
                        num_samples = min(mid, Xc_test.shape[0])
                        model.fit(Xc_test[:num_samples], Yc_test[:num_samples],
                                 batch_size=mid, epochs=1, verbose=0)
                        tee("✓ fits")
                        low = mid
                    except (tensorflow.errors.ResourceExhaustedError,
                            tensorflow.errors.InternalError):
                        tee("✗ OOM")
                        high = mid

            # Clean up
            tensorflow.keras.backend.clear_session()
            return low

        optimal_batch = find_max_batch_size(DEFAULT_BATCH_SIZE)
        # Use 80% of max to leave headroom for variance across batches
        BATCH_SIZE = int(optimal_batch * 0.8)
        # Round to multiple of N_GPUS
        BATCH_SIZE = (BATCH_SIZE // N_GPUS) * N_GPUS

        tee(f"\n\033[1m✓ Auto-tuned batch size: {BATCH_SIZE} ({BATCH_SIZE // N_GPUS} per GPU)\033[0m")
        tee(f"  (Using 80% of max {optimal_batch} for safety margin)\n")

    num_batches = args.num_epochs*len(idx_train)

    start_time = time.time()

    def print_performance_metrics(indices):
        Y_true = []
        for i in range(6):
            Y_true.append([[] for t in range(1)])
        Y_pred = []
        for i in range(6):
            Y_pred.append([[] for t in range(1)])

        for idx in indices:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS, CL_max)
            Yp = model.predict(Xc, batch_size=BATCH_SIZE)

            if not isinstance(Yp, list):
                Yp = [Yp]

            t = 0
            is_expr = (Yc[t].sum(axis=(1, 2)) >= 1)

            for i in range(6):
                Y_true[i][t].extend(Yc[t][is_expr, :, i].flatten())
                Y_pred[i][t].extend(Yp[t][is_expr, :, i].flatten())

            tee("\n\033[1mAcceptor:\033[0m")
            acceptor_score = print_topl_statistics(np.asarray(Y_true[1][t]), np.asarray(Y_pred[1][t]), print_fn=tee)

            tee("\n\033[1mDonor:\033[0m")
            donor_score = print_topl_statistics(np.asarray(Y_true[2][t]), np.asarray(Y_pred[2][t]), print_fn=tee)

            tee("\n\033[1mCDS start:\033[0m")
            print_topl_statistics(np.asarray(Y_true[3][t]), np.asarray(Y_pred[3][t]), print_fn=tee)

            tee("\n\033[1mCDS end:\033[0m")
            print_topl_statistics(np.asarray(Y_true[4][t]), np.asarray(Y_pred[4][t]), print_fn=tee)

            tee("\n\033[1mis_exon:\033[0m")
            print_basic_statistics(np.asarray(Y_true[5][t]), np.asarray(Y_pred[5][t]), print_fn=tee)

            return acceptor_score, donor_score

    for batch_idx in range(num_batches):

        idx = np.random.choice(idx_train)

        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]

        Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS, CL_max)
        model.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)

        if (batch_idx+1) % len(idx_train) == 0:
            epoch_num = (batch_idx+1) // len(idx_train)
            # Printing metrics (see utils.py for details)

            tee("--------------------------------------------------------------")
            tee("epoch_num:", epoch_num)
            tee("\n\033[1mValidation set metrics:\033[0m")
            acceptor_score, donor_score = print_performance_metrics(idx_valid)

            tee("\n\033[1mTraining set metrics:\033[0m")
            print_performance_metrics(idx_train[:len(idx_valid)])

            from tensorflow.python.keras import backend
            K = backend
            tee("Learning rate: %.5f" % (K.get_value(model.optimizer.learning_rate)))
            tee("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

            tee("--------------------------------------------------------------")
            sys.stdout.flush()

            filename = ('GeneML' + str(args.context_length) + '_c' + args.dataset_name + f'_ep{epoch_num}.h5')
            tee.f.flush()

            os.makedirs(args.keras_save_path, exist_ok=True)
            local_path_keras = os.path.join(args.keras_save_path, filename.replace('.h5', '.keras'))
            model.save(local_path_keras)

            if epoch_num >= 6:
                K.set_value(model.optimizer.learning_rate, 0.5 * K.get_value(model.optimizer.learning_rate))
                # Learning rate decay

            if acceptor_score < 0.1 and donor_score < 0.1:
                tee('Weak training--exiting early')
                sys.exit(0)

    h5f.close()


if __name__ == '__main__':
    main()
