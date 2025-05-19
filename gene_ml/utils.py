import numpy as np

from gene_ml.model_loader import ResidualModelBase


def chunked_seq_predict(model: ResidualModelBase, seq: str, chunk_size=100000, padding=1000):
    """
    Predicts the sequence in chunks of a given size to control memory usage, with some padding to handle
    the sequence context
    """
    seq = seq.upper()
    seq_len = len(seq)
    pred_list = []
    for i in range(0, seq_len, chunk_size):
        start = i
        end = min(seq_len, i + chunk_size)
        padded_start = max(0, start - padding)
        padded_end = min(seq_len, end + padding)
        preds = model.predict(seq[padded_start:padded_end], return_dict=False)
        pred_list.append(preds[:, (start-padded_start):(end-padded_start)])
    return np.concatenate(pred_list, axis=1)
