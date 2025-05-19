import os
import re

import numpy as np
from keras.src import backend
from keras.src.saving import register_keras_serializable

MODEL_EXON_START = 1
MODEL_EXON_END = 2
MODEL_CDS_START = 3
MODEL_CDS_END = 4
MODEL_IS_EXON = 5
MODEL_IS_INTRON = 6


class ResidualModelBase:
    default_path = None

    def __init__(self, path=None):
        self.specify_model_parameters()

        if path is None:
            path = self.default_path
        print(f"Loading model from {path}")

        from keras.src.saving import load_model  # keras v3
        self.model = load_model(path)

        filename = os.path.basename(path)
        # infer context length from model name
        match = re.search(r'\d+', filename)
        if match:
            self.context_length = int(match.group(0))
        else:
            raise ValueError(f"Could not extract context length from filename: {filename}")

    def predict(self, seq, return_dict=True):
        x = self._one_hot_encode('N'*(self.context_length//2) + seq.upper() + 'N'*(self.context_length//2))[None, :]
        y = self.model.predict(x, verbose=0)

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

        return map[np.fromstring(seq, np.int8) % 5]

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
        # self.default_path = 'gs://hx-lawrence/geneml-jobs/Models/geneml_20250517_215basidio_jgigenepred_v10/GeneML800_c215basidio_jgigenepred_ep10.keras'
        # self.default_path = 'gs://hx-lawrence/geneml-jobs/Models/geneml_20250517_308ascomycota_jgigenepred_v2/GeneML800_c308ascomycota_jgigenepred_ep10.keras'
        self.default_path = 'gs://hx-lawrence/geneml-jobs/Models/geneml_20250517_726g_jgigenepred_v4/GeneML800_c726g_jgigenepred_ep10.keras'

        self.annotations = [
            'none',
            'exon_start',  # splice acceptor
            'exon_end',  # splice donor
            'cds_start',
            'cds_end',
            'is_exon',
            'is_intron',
        ]


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
