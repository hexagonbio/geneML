import os
from functools import cache

import numpy as np
from keras.src import backend
from keras.src.saving import load_model, register_keras_serializable

MODEL_EXON_START = 1
MODEL_EXON_END = 2
MODEL_CDS_START = 3
MODEL_CDS_END = 4
MODEL_IS_EXON = 5
MODEL_IS_INTRON = 6

DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/geneML_default.keras"))

class ResidualModelBase:
    def __init__(self, path, context_length):
        self.specify_model_parameters()

        self.model = load_model(path)
        self.context_length = context_length

    def predict(self, seq, return_dict=True):
        x = self._one_hot_encode('N'*(self.context_length//2) + seq.upper() + 'N'*(self.context_length//2))[None, :]
        y = self.model([x], training=False).numpy()

        if return_dict:
            return {annot: y[0, :, i] for i, annot in enumerate(self.annotations)}
        else:
            return y[0].T

    def _one_hot_encode(self, seq):
        """ version from the spliceai package """

        map = np.asarray([[0, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
        seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

        return map[np.frombuffer(seq.encode('ascii'), np.int8) % 5]

    def specify_model_parameters(self):
        """
        This method should be overridden by subclasses to specify model parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class ExonIntron6ClassModel(ResidualModelBase):
    """
    Residual model with an expanded number of classes trained on genes plus some sequence context outside of genes
    """
    def specify_model_parameters(self):
        self.annotations = [
            'none',
            'exon_start',  # splice acceptor
            'exon_end',  # splice donor
            'cds_start',
            'cds_end',
            'is_exon',
            'is_intron',
        ]


@cache
def get_cached_gene_ml_model(model_path, context_length):
    if not model_path:
        model_path = DEFAULT_MODEL_PATH
    return ExonIntron6ClassModel(path=model_path, context_length=context_length)


@register_keras_serializable()
def categorical_crossentropy_2d_gene_ml(y_true, y_pred):
    # Standard categorical cross entropy for sequence outputs
    kb = backend
    return - kb.mean(y_true[:, :, 0]*kb.log(y_pred[:, :, 0]+1e-10)
                     + y_true[:, :, 1]*kb.log(y_pred[:, :, 1]+1e-10)
                     + y_true[:, :, 2]*kb.log(y_pred[:, :, 2]+1e-10)
                     + y_true[:, :, 3]*kb.log(y_pred[:, :, 3]+1e-10)
                     + y_true[:, :, 4]*kb.log(y_pred[:, :, 4]+1e-10)
                     + y_true[:, :, 5]*kb.log(y_pred[:, :, 5]+1e-10)
                     + y_true[:, :, 6]*kb.log(y_pred[:, :, 6]+1e-10))
