import os
import re
from functools import cache

import numpy as np

from tensorflow.python.keras.models import load_model
# from tf_keras.models import load_model



class ResidualModelBase:
    default_path = None

    def __init__(self, path=None):
        self.specify_model_parameters()

        if path is None:
            path = self.default_path

        filename = os.path.basename(path)

        self.model = load_model(path)  #.expect_partial()

        if path.endswith('.h5'):
            # infer context length from model name
            match = re.search(r'\d+', filename)
            if match:
                self.context_length = int(match.group(0))
            else:
                raise ValueError(f"Could not extract context length from filename: {filename}")
        else:
            self.context_length = 800

    def predict(self, seq, return_dict=True):
        x = self._one_hot_encode('N'*(self.context_length//2) + seq.upper() + 'N'*(self.context_length//2))[None, :]
        y = self.model.predict(x, verbose=0)

        if return_dict:
            return {annot: y[0, :, i] for i, annot in enumerate(self.annotations)}
        else:
            return y

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
        # self.default_path = 'gs://hx-lawrence/spliceai-jobs/Models/spliceai_20210218_332g_dassei800cl_v3/SpliceAI800_c332g_dassei800cl.h5'
        self.default_path = 'gs://hx-lawrence/geneml-jobs/Models/geneml_20250515_726g_jgigenepred_v1/tf_model_ep3/'

        self.annotations = [
            'none',
            'exon_start',  # splice acceptor
            'exon_end',  # splice donor
            'cds_start',
            'cds_end',
            'is_exon',
            'is_intron',
        ]

        # from gene_ml.loss_functions import categorical_crossentropy_2d_gene_ml
        # self.loss_function = categorical_crossentropy_2d_gene_ml




@cache
def get_cached_gene_ml_model(path=None):
    return ExonIntron6ClassModel(path)
