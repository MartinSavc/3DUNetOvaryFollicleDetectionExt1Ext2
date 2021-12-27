import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Activation, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from unet3d.metrics import dice_coeff, dice_loss, bce_dice_rho_loss
from unet3d.model.unet import get_up_convolution

def iss_unet_model_3d(input_shape, preproc_kernel_size=(5, 5, 5), pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False, depth=4, n_filters=128, metrics=dice_coeff, batch_normalization=False, activation_name="sigmoid"):
    """
    Builds a interscale similar 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param preproc_kernel_size: size of the preprocessing kernel applied to the input to transform it before the first layer.
    :param n_filters: The number of filters used on each layer.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling layers will be added to the model. 
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
        inputs = Input(input_shape)
    else:
        if K.image_data_format() == 'channels_first':
            inputs = Input((1,None,None,None))
        else:
            inputs = Input((None,None,None,1))

    preproc_conv_layer = Conv3D(n_filters, preproc_kernel_size, padding='same', activation='relu')

    # down-step layers
    conv_layer_1 = create_convolution_block_layer(n_filters=n_filters, batch_normalization=batch_normalization)
    conv_layer_2 = create_convolution_block_layer(n_filters=n_filters, batch_normalization=batch_normalization)
    maxpool_layer = MaxPooling3D(pool_size=pool_size)

    # up-step layers
    upconv_layer = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=n_filters)
    if K.image_data_format() == 'channels_first':
        concat_layer = Concatenate(axis=1)
    else:
        concat_layer = Concatenate(axis=-1)
    conv_layer_3 = create_convolution_block_layer(n_filters=n_filters, batch_normalization=batch_normalization)
    conv_layer_4 = create_convolution_block_layer(n_filters=n_filters, batch_normalization=batch_normalization)

    # processing down
    current_tens = preproc_conv_layer(inputs)
    levels = list()
    for layer_depth in range(depth):
        tens1 = conv_layer_1(current_tens)
        tens2 = conv_layer_2(tens1)
        if layer_depth < depth - 1:
            current_tens = maxpool_layer(tens2)
            levels.append([tens1, tens2, current_tens])
        else:
            current_tens = tens2
            levels.append([tens1, tens2])

    # processing up
    for layer_depth in range(depth-2, -1, -1):
        up_tens = upconv_layer(current_tens)
        concat_tens = concat_layer([up_tens, levels[layer_depth][1]])
        current_tens = conv_layer_3(concat_tens)
        current_tens = conv_layer_4(current_tens)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_tens)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=bce_dice_rho_loss, metrics=metrics)
    return model


def create_convolution_block_layer(n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None, padding='same', strides=(1, 1, 1), instance_normalization=False):
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
    conv_layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)

    if batch_normalization:
        if K.image_data_format() == 'channels_first':
            norm_layer = BatchNormalization(axis=1)
        else:
            norm_layer = BatchNormalization(axis=-1)
    elif instance_normalization:
        try:
            from keras_contrib.layers import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        if K.image_data_format() == 'channels_first':
            norm_layer = InstanceNormalization(axis=1)
        else:
            norm_layer = InstanceNormalization(axis=-1)

    else:
        norm_layer = lambda x: x

    if activation is None:
        activation_layer = Activation('relu')
    else:
        activation_layer = activation()

    def layer_fun(input_tens):
        tens = conv_layer(input_tens)
        tens = norm_layer(tens)
        tens = activation_layer(tens)
        return tens

    return layer_fun
