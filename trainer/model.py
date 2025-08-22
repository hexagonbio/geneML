###############################################################################
# This file has the functions necessary to create the GeneML model.
###############################################################################

import numpy as np

# hxenv
# from tf_keras.layers import Input
# from tf_keras.layers import Conv1D, Cropping1D
# from tf_keras.layers import Activation
# from tf_keras.layers import add
# from tf_keras.layers import BatchNormalization
# from tf_keras.models import Model
# ai-platform
from keras.layers import Activation, BatchNormalization, Conv1D, Cropping1D, Input, add
from keras.models import Model

# from tensorflow.keras.layers import VersionAwareLayers
# layers = VersionAwareLayers()
# BatchNormalization = layers.BatchNormalization


def ResidualUnit(ll, w, ar):
    # Residual unit proposed in "Identity mappings in Deep Residual Networks"
    # by He et al.

    def f(input_node):

        bn1 = BatchNormalization()(input_node)
        act1 = Activation('relu')(bn1)
        conv1 = Conv1D(ll, w, dilation_rate=ar, padding='same')(act1)
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv1D(ll, w, dilation_rate=ar, padding='same')(act2)
        output_node = add([conv2, input_node])

        return output_node

    return f


def GeneML(L, W, AR, num_classes):
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit

    assert len(W) == len(AR), 'failed: len(W) == len(AR)'

    CL = 2 * np.sum(AR*(W-1))

    input0 = Input(shape=(None, 4))
    conv = Conv1D(L, 1)(input0)
    skip = Conv1D(L, 1)(conv)

    for i in range(len(W)):
        conv = ResidualUnit(L, [W[i]], [AR[i]])(conv)

        if ((i+1) % 4 == 0) or ((i+1) == len(W)):
            # Skip connections to the output after every 4 residual units
            dense = Conv1D(L, 1)(conv)
            skip = add([skip, dense])

    skip = Cropping1D(int(CL/2))(skip)

    output0 = [[] for _ in range(1)]

    for t in range(1):
        output0[t] = Conv1D(num_classes, 1, activation='softmax')(skip)

    model = Model(inputs=input0, outputs=output0)

    return model
