###############################################################################
# This file contains the code to train the GeneML model.
###############################################################################

import argparse
import gc
import os
import re
import sys
import time


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
        '--max-eval-samples',
        type=int,
        default=256,
        help='Maximum number of samples to evaluate metrics on per split (default: %(default)s)',
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=1,
        help='Run evaluation every N epochs (default: %(default)s).' \
             'Always evaluates on first and last epochs.',
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=48,
        help='Batch size to use for evaluation/prediction (default: %(default)s). Smaller values reduce GPU memory during eval.',
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=2,
        help='Stop training if validation metric does not improve for N consecutive evaluations (default: %(default)s). Set to 0 to disable.',
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=0.0005,
        help='Minimum improvement in validation metric to reset patience (default: %(default)s)',
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    os.environ["TF_USE_LEGACY_KERAS"] = "0"  # use keras v3 even though we have tf_keras installed

    import random

    import h5py
    import keras
    import numpy as np
    import tensorflow

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tensorflow.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # works locally and on Vertex AI
    from model import GeneML
    from utils import print_basic_statistics, print_topl_statistics


    @keras.saving.register_keras_serializable()
    class WarmupSchedule(tensorflow.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, start_lr, target_lr, warmup_steps):
            self.start_lr = start_lr
            self.target_lr = target_lr
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tensorflow.cast(step, tensorflow.float32)
            return tensorflow.cond(
                step < self.warmup_steps,
                lambda: self.start_lr + (self.target_lr - self.start_lr) * step / self.warmup_steps,
                lambda: self.target_lr
            )

        def get_config(self):
            return {
                'start_lr': self.start_lr,
                'target_lr': self.target_lr,
                'warmup_steps': self.warmup_steps,
            }


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
    SL = int(os.path.basename(args.train_path).split('_')[3])

    ###############################################################################
    # Model
    ###############################################################################

    # Hyper-parameters:
    L = 32      # Number of convolution kernels
    W = None    # Convolution window size in each residual unit
    AR = None   # Atrous rate in each residual unit

    N_GPUS = args.num_gpus
    BATCH_SIZE = None   # Number of sequences per training step

    LEARNING_RATE = None    # Final learning rate
    WARMUP_EPOCHS = None    # Number of epochs for learning rate warmup
    WEIGHT_DECAY = 0.0001   # Weight decay for Adam optimizer


    if int(args.context_length) == 800:
        W =  np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1,  1,  1,  1,  4,  4,  4,  4,  10, 10])
        BATCH_SIZE = 72*N_GPUS
        LEARNING_RATE = 0.0003
        WARMUP_EPOCHS = 0.5

    elif int(args.context_length) == 1400:
        W  = np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 15, 15])
        AR = np.asarray([1,  1,  1,  1,  4,  4,  4,  4,  8,  8])
        BATCH_SIZE = 60*N_GPUS
        LEARNING_RATE = 0.00025
        WARMUP_EPOCHS = 0.75

    elif int(args.context_length) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                         10, 10, 10, 10])
        BATCH_SIZE = 48*N_GPUS
        LEARNING_RATE = 0.0002
        WARMUP_EPOCHS = 1

    else:
        assert False, 'failed: int(args.context_length) does not match any known configuration'

    CL = 2 * np.sum(AR*(W-1))
    assert CL == int(args.context_length), 'Context length of model (%d) does not match argument (%d)' % (CL, int(args.context_length))
    tee("\033[1mContext nucleotides: %d\033[0m" % (CL))
    tee("\033[1mSequence length (output): %d\033[0m" % (SL))

    @keras.saving.register_keras_serializable()
    def categorical_crossentropy_2d_gene_ml(y_true, y_pred):
        kb = keras.ops

        # Mask for non-padding positions (sum over classes > 0)
        mask = kb.sum(y_true, axis=-1) > 0
        mask = kb.cast(mask, y_pred.dtype)

        loss_per_pos = -kb.sum(y_true * kb.log(y_pred + 1e-10), axis=-1)
        masked_loss = loss_per_pos * mask
        loss = kb.sum(masked_loss) / (kb.sum(mask) + 1e-8)
        return loss

    num_classes = 7
    loss = categorical_crossentropy_2d_gene_ml

    h5f = h5py.File(args.train_path, 'r')

    num_idx = len(h5f.keys())//2
    tee('num_idx:', num_idx)
    num_idx_train = int(0.9 * num_idx)

    # Set learning rate schedule with warmup
    steps_per_epoch = num_idx_train // BATCH_SIZE
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    lr_schedule = WarmupSchedule(
        start_lr=LEARNING_RATE * 0.01,
        target_lr=LEARNING_RATE,
        warmup_steps=warmup_steps,
    )

    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    # Enable GPU memory growth to avoid pre-allocating all VRAM
    try:
        gpus = tensorflow.config.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    if N_GPUS > 1:
        # https://keras.io/guides/distributed_training/
        strategy = tensorflow.distribute.MirroredStrategy()
        tee('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = GeneML(L, W, AR, num_classes)
            model.compile(loss=loss,
                          optimizer=keras.optimizers.Adam(learning_rate=lr_schedule,
                                                          weight_decay=WEIGHT_DECAY))
    else:
        model = GeneML(L, W, AR, num_classes)
        model.compile(loss=loss,
                      optimizer=keras.optimizers.Adam(learning_rate=lr_schedule,
                                                      weight_decay=WEIGHT_DECAY))
    # model.summary()

    ###############################################################################
    # Training and validation
    ###############################################################################

    idx_all = np.random.permutation(num_idx)
    idx_train = idx_all[:int(0.9*num_idx)]
    idx_valid = idx_all[int(0.9*num_idx):]

    # Use a fixed validation subset across all epochs for stable metrics
    eval_count = min(len(idx_valid), args.max_eval_samples)
    val_eval_indices = idx_valid[:eval_count]
    # Use a fixed training subset of the same size for comparable reporting
    train_eval_indices = idx_train[:eval_count]

    start_time = time.time()

    # Early stopping tracking
    best_val_score = -1.0
    best_epoch = 0
    patience_counter = 0

    # Evaluation batch size can be smaller than training to reduce peak GPU mem
    EVAL_BATCH_SIZE = max(1, min(BATCH_SIZE, args.eval_batch_size * N_GPUS))

    def print_performance_metrics(indices, max_eval):
        # Cap evaluated samples to avoid excessive memory use during metrics
        if len(indices) > max_eval:
            indices = np.random.choice(indices, max_eval, replace=False)

        Y_true = []
        for i in range(7):
            Y_true.append([[] for t in range(1)])
        Y_pred = []
        for i in range(7):
            Y_pred.append([[] for t in range(1)])

        for idx in indices:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = X, [Y[0]]
            Yp = model.predict(Xc, batch_size=EVAL_BATCH_SIZE, verbose=0)

            if not isinstance(Yp, list):
                Yp = [Yp]

            t = 0
            is_expr = (Yc[t].sum(axis=(1, 2)) >= 1)

            for i in range(7):
                Y_true[i][t].extend(Yc[t][is_expr, :, i].flatten())
                Y_pred[i][t].extend(Yp[t][is_expr, :, i].flatten())

            # Explicitly free large arrays from memory
            del X, Y, Xc, Yc, Yp

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

        tee("\n\033[1mis_intron:\033[0m")
        print_basic_statistics(np.asarray(Y_true[6][t]), np.asarray(Y_pred[6][t]), print_fn=tee)

        # Explicitly free large metric arrays
        del Y_true, Y_pred

        return acceptor_score, donor_score

    for epoch_num in range(1, args.num_epochs + 1):
        # Shuffle indices for this epoch (no replacement)
        # Use epoch-based seed for deterministic but different permutation per epoch
        np.random.seed(SEED + epoch_num)
        epoch_indices = np.random.permutation(idx_train)

        for idx in epoch_indices:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = X, [Y[0]]
            model.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)

        is_first_epoch = (epoch_num == 1)
        is_last_epoch = (epoch_num == args.num_epochs)
        should_eval = is_first_epoch or is_last_epoch or ((epoch_num - 1) % args.eval_every == 0)
        tee("--------------------------------------------------------------")
        tee("epoch_num:", epoch_num)

        if should_eval:
            # Printing metrics (see utils.py for details)
            tee("\n\033[1mValidation set metrics:\033[0m")
            acceptor_score, donor_score = print_performance_metrics(val_eval_indices, len(val_eval_indices))

            tee("\n\033[1mTraining set metrics:\033[0m")
            print_performance_metrics(train_eval_indices, len(train_eval_indices))

            # Early stopping check based on average of acceptor and donor Top-1L
            current_val_score = (acceptor_score + donor_score) / 2.0
            if current_val_score > best_val_score + args.early_stopping_min_delta:
                best_val_score = current_val_score
                best_epoch = epoch_num
                patience_counter = 0
                tee(f"\033[92m*** New best validation score: {best_val_score:.4f} (epoch {best_epoch}) ***\033[0m")
            else:
                patience_counter += 1
                tee(f"No improvement for {patience_counter} evaluation(s). Best: {best_val_score:.4f} (epoch {best_epoch})")

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

            # Clear memory after metrics printing
            gc.collect()
            tensorflow.keras.backend.clear_session()

            if acceptor_score < 0.1 and donor_score < 0.1:
                tee('Weak training--exiting early')
                sys.exit(0)

            # Early stopping: halt if patience exceeded
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                tee(f"\n\033[93mEarly stopping triggered after {patience_counter} evaluations without improvement.\033[0m")
                tee(f"\033[92mBest epoch: {best_epoch} with validation score: {best_val_score:.4f}\033[0m")
                tee(f"\033[92mRecommended checkpoint: GeneML{args.context_length}_c{args.dataset_name}_ep{best_epoch}.keras\033[0m")
                h5f.close()
                sys.exit(0)
        else:
            tee(f"Skipping evaluation this epoch (eval_every={args.eval_every})")
            tee("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            tee("--------------------------------------------------------------")
            sys.stdout.flush()

    # Training completed without early stopping
    if args.early_stopping_patience > 0:
        tee(f"\n\033[92mTraining completed. Best epoch: {best_epoch} with validation score: {best_val_score:.4f}\033[0m")
        tee(f"\033[92mRecommended checkpoint: GeneML{args.context_length}_c{args.dataset_name}_ep{best_epoch}.keras\033[0m")

    h5f.close()


if __name__ == '__main__':
    main()
