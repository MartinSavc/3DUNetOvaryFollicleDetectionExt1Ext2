import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
# from keras.engine import Input, Model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose
from classes_3D import DataGenerator

from tensorflow.keras.optimizers import Adam

from unet3d.metrics import dice_coeff, dice_loss, bce_dice_rho_loss

K.set_image_data_format("channels_first")

# try:
#     from keras.engine import merge
# except ImportError:
#     from keras.layers.merge import concatenate

try:
    from keras.engine import merge
except ImportError:
    from tensorflow.keras.layers import concatenate

class DataGeneratorMod(DataGenerator):
    def __init__(self, unet_levels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unet_levels = unet_levels

    def __getitem__(self, index):
        volume, label = super().__getitem__(index)
        labels_list = [label]
        for l in range(1, self.unet_levels):
            B, C, H, W, D = label.shape
            label = label.reshape(B, C, H//2, 2, W//2, 2, D//2, 2).mean((3, 5, 7))
            labels_list.insert(0, label)
        return volume, labels_list


def unet_guided_model_3d(input_shape, pool_size=(2, 2, 2), kernel_size=(3, 3, 3), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, metrics=dice_coeff, batch_normalization=False, activation_name="sigmoid", loss_weights=None, build_model=True):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    if input_shape is not None:
        try:
            if K.is_keras_tensor(input_shape):
                inputs=input_shape
            else:
                inputs = Input(input_shape)
        except ValueError:
            inputs = Input(input_shape)
    else:
        inputs = Input((1,None,None,None))
    
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth), kernel=kernel_size,
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2, kernel=kernel_size,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    outputs_list = []
    logits_layer = Conv3D(n_labels, (1, 1, 1))(current_layer)
    output_layer = Activation(activation_name)(logits_layer)
    outputs_list.append(output_layer)

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer.shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1].shape[1],
                                                 kernel=kernel_size,
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1].shape[1],
                                                 kernel=kernel_size,
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)
        logits_layer = Conv3D(n_labels, (1, 1, 1))(current_layer)
        if layer_depth == 0:
            uid = K.get_uid('final_output')-1
            if uid == 0:
                final_output_name = 'final_output'
            else:
                final_output_name = f'final_output_{uid+1}'
            output_layer = Activation(activation_name, name=final_output_name)(logits_layer)
        else:
            output_layer = Activation(activation_name)(logits_layer)
        outputs_list.append(output_layer)

    #final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    #act = Activation(activation_name)(final_convolution)
    if not build_model:
        return outputs_list
    else:
        model = Model(inputs=inputs, outputs=outputs_list)

        if not isinstance(metrics, list):
            metrics = [metrics]

        #    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=weighted_dice_coefficient_loss, metrics=metrics)

        if loss_weights is None:
            loss_wights = [1.]*len(outputs_list)

        model.compile(optimizer=Adam(lr=initial_learning_rate), 
                loss=[bce_dice_rho_loss]*len(outputs_list),
                metrics={'final_output':metrics},
                loss_weights=loss_weights,
                )
        return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                                strides=strides)
    else:
        return UpSampling3D(size=pool_size)
