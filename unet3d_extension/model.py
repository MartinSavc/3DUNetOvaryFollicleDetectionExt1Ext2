import tensorflow as tf
from unet3d.model.unet_guided import unet_guided_model_3d
from unet3d.model.unet import unet_model_3d, create_convolution_block, get_up_convolution

def unet_extension_model_3d(base_model, n_labels, ext_type='ext1'):
    for l in base_model.layers:
        l._name = 'base/'+l._name
    if ext_type=='ext1':
        return unet_ext1(base_model, n_labels)
    if ext_type=='ext2':
        return unet_ext2(base_model, n_labels)
    else:
        raise Exception(f'Not a valid extension type: {ext_type}')

def get_model_params(model):
    base_output_tens_list = model.outputs
    base_output_tens = base_output_tens_list[-1]

    depth=len([l for l in model.layers if isinstance(l, tf.keras.layers.MaxPooling3D)])
    if depth==0:
        raise Exception('Unknown model, found no MaxPooling3D layers')
    depth+=1

    conv1_ind=1
    maxpool_ind=7
    upsample_ind=-10
    conv2_ind=-2
    act_ind=-1


    if len(base_output_tens_list) > 1:
        guided=True
        conv2_ind-=(depth-1)
        upsample_ind-=(depth-1)*2
        loss_weights=model._get_compile_args()['loss_weights']
    else:
        guided=False
        loss_weights=None

    if isinstance(model.layers[2], tf.keras.layers.BatchNormalization):
        batch_normalization=True
    else:
        batch_normalization=False
        maxpool_ind-=2
        upsample_ind+=2

    if not isinstance(model.layers[conv1_ind], tf.keras.layers.Conv3D):
        raise Exception('Unknown model, expected Conv3D, found {type(model.layers[conv1_ind])}')
    if not isinstance(model.layers[maxpool_ind], tf.keras.layers.MaxPooling3D):
        raise Exception('Unknown model, expected MaxPooling3D, found {type(model.layers[maxpool_ind])}')
    if not isinstance(model.layers[conv2_ind], tf.keras.layers.Conv3D):
        raise Exception('Unknown model, expected Conv3D, found {type(model.layers[conv2_ind])}')
    if not isinstance(model.layers[act_ind], tf.keras.layers.Activation):
        raise Exception('Unknown model, expected Activation, found {type(model.layers[act_ind])}')

    if isinstance(model.layers[upsample_ind], tf.keras.layers.Conv3DTranspose):
        deconvolution=True
    elif isinstance(model.layers[upsample_ind], tf.keras.layers.UpSampling3D):
        deconvolution=False
    else:
        raise Exception('Unknown model, expected Conv3DTranspose or UpSampling3D, found {type(model.layers[upsample_ind])}')

    pool_size=model.layers[maxpool_ind].pool_size
    kernel_size=model.layers[conv1_ind].kernel_size
    dilation=model.layers[conv1_ind].dilation_rate
    n_base_filters=model.layers[conv1_ind].filters
    activation_name=tf.keras.activations.serialize(model.layers[act_ind].activation)

    return {
            'depth' : depth,
            'pool_size' : pool_size,
            'kernel_size' : kernel_size,
            'n_base_filters' : n_base_filters,
            'dilation' : dilation,
            'activation_name' : activation_name,
            'batch_normalization' : batch_normalization,
            'guided' : guided,
            'loss_weights' : loss_weights,
            'deconvolution' : deconvolution,
            }

def unet_ext1(base_model, n_labels):
    input_tens, = base_model.inputs
    base_output_tens_list = base_model.outputs
    base_output_tens = base_output_tens_list[-1]

    model_params = get_model_params(base_model)
    pool_size = model_params['pool_size']
    kernel_size = model_params['kernel_size']
    deconvolution = model_params['deconvolution']
    dilation = model_params['dilation']
    depth = model_params['depth']
    n_base_filters = model_params['n_base_filters']
    batch_normalization = model_params['batch_normalization']
    activation_name = model_params['activation_name']
    loss_weights = model_params['loss_weights']
    guided = model_params['guided']

    ext_input_tens = tf.keras.layers.Concatenate(axis=1)([input_tens, base_output_tens])

    if guided:
        ext_output_tens_list = unet_guided_model_3d(ext_input_tens,
                pool_size=pool_size,
                kernel_size=kernel_size,
                n_labels=n_labels,
                deconvolution=deconvolution,
                depth=depth,
                n_base_filters=n_base_filters,
                batch_normalization=batch_normalization,
                activation_name=activation_name,
                loss_weights=loss_weights,
                build_model=False,
                )
    else:
        ext_output_tens_list = unet_model_3d(ext_input_tens,
                pool_size=pool_size,
                kernel_size=kernel_size,
                n_labels=n_labels,
                deconvolution=deconvolution,
                dilation=dilation,
                depth=depth,
                n_base_filters=n_base_filters,
                batch_normalization=batch_normalization,
                activation_name=activation_name,
                build_model=False,
                )

    output_tens_list = []
    for base_out_tens, ext_out_tens in zip(base_output_tens_list[:-1], ext_output_tens_list[:-1]):
        output_tens = tf.keras.layers.Concatenate(axis=1)([ext_out_tens, base_out_tens])
        output_tens_list.append(output_tens)

    uid = tf.keras.backend.get_uid('final_output')-1
    if uid == 0:
        final_output_name = f'final_output'
    else:
        final_output_name = f'final_output_{uid+1}'
    output_tens = tf.keras.layers.Concatenate(axis=1, name = final_output_name)([ext_output_tens_list[-1], base_output_tens_list[-1]])
    output_tens_list.append(output_tens)

    model = tf.keras.Model(inputs=input_tens, outputs=output_tens_list)
    comp_args = base_model._get_compile_args()
    if guided:
        if isinstance(comp_args['metrics'], dict):
            comp_args['metrics'] = {final_output_name: comp_args['metrics']['final_output']}
        else:
            comp_args['metrics'] = {final_output_name: comp_args['metrics']}
    model.compile(**comp_args)
    return model



def unet_ext2(base_model, n_labels):
    input_tens, = base_model.inputs
    base_output_tens_list = base_model.outputs
    base_output_tens = base_output_tens_list[-1]

    model_params = get_model_params(base_model)
    guided = model_params['guided']
    depth = model_params['depth']
    kernel_size = model_params['kernel_size']
    dilation = model_params['dilation']
    batch_normalization = model_params['batch_normalization']
    pool_size = model_params['pool_size']
    deconvolution = model_params['deconvolution']
    activation_name = model_params['activation_name']
    loss_weights = model_params['loss_weights']

    level_feature_maps = [l.input for l in base_model.layers if isinstance(l, (tf.keras.layers.Conv3DTranspose, tf.keras.layers.UpSampling3D))]

    if guided:
        final_conv_layer = base_model.layers[-depth-1]
    else:
        final_conv_layer = base_model.layers[-2]

    if not isinstance(final_conv_layer, tf.keras.layers.Conv3D):
        raise Exception('Unknown model, expected Conv3D, found {type(final_conv_layer)}')
    level_feature_maps.append(final_conv_layer.input)

    ext_output_tens_list = []
    feat_map_upsample = None
    for l in range(depth):
        feat_map = level_feature_maps[l]
        n_filters = feat_map.shape[1]

        if feat_map_upsample is not None:
            feat_map = tf.keras.layers.Concatenate(axis=1)([feat_map_upsample, feat_map])

        feat_map = create_convolution_block(n_filters=n_filters,
                                                 kernel=kernel_size,
                                                 input_layer=feat_map,
                                                 batch_normalization=batch_normalization)

        feat_map = create_convolution_block(n_filters=n_filters,
                                                 kernel=kernel_size,
                                                 input_layer=feat_map,
                                                 dilation=dilation,
                                                 batch_normalization=batch_normalization)

        if l==depth-1:
            break

        feat_map_upsample = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=n_filters)(feat_map)

        if guided:
            logits = tf.keras.layers.Conv3D(n_labels, (1, 1, 1))(feat_map)
            ext_output = tf.keras.layers.Activation(activation_name)(logits)
            ext_output_tens_list.append(ext_output)

    logits = tf.keras.layers.Conv3D(n_labels, (1, 1, 1))(feat_map)
    ext_output = tf.keras.layers.Activation(activation_name)(logits)
    ext_output_tens_list.append(ext_output)


    output_tens_list = []
    for base_out_tens, ext_out_tens in zip(base_output_tens_list[:-1], ext_output_tens_list[:-1]):
        output_tens = tf.keras.layers.Concatenate(axis=1)([ext_out_tens, base_out_tens])
        output_tens_list.append(output_tens)


    uid = tf.keras.backend.get_uid('final_output')-1
    if uid == 0:
        final_output_name = f'final_output'
    else:
        final_output_name = f'final_output_{uid+1}'
    output_tens = tf.keras.layers.Concatenate(axis=1, name = final_output_name)([ext_output_tens_list[-1], base_output_tens_list[-1]])
    output_tens_list.append(output_tens)

    model = tf.keras.Model(inputs=input_tens, outputs=output_tens_list)
    comp_args = base_model._get_compile_args()
    if guided:
        if isinstance(comp_args['metrics'], dict):
            comp_args['metrics'] = {final_output_name: comp_args['metrics']['final_output']}
        else:
            comp_args['metrics'] = {final_output_name: comp_args['metrics']}
    model.compile(**comp_args)
            #optimizer=base_model.optimizer,
            #loss=base_model.loss,
            #metrics=base_model.metrics,
            #loss_weights=loss_weights,
            #)
    return model
