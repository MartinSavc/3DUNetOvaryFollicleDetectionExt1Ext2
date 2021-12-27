import tensorflow as tf

class IsometricSeparableConv3D(tf.keras.layers.Layer):
    def __init__(self,
            filters,
            kernel_size,
            strides=1,
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if self.data_format is None:
            self.data_format = tf.keras.backend.image_data_format()
        if self.data_format == 'channels_first':
            self.channel_axis=1
        else:
            self.channel_axis=-1

        self.input_dim = None

    def get_config(self):
        base_conf = super().get_config()
        conf_dict = {
                'filters':self.filters,
                'kernel_size':self.kernel_size,
                'strides':self.strides,
                'data_format':self.data_format,
                'dilation_rate':self.dilation_rate,
                'activation':tf.keras.activations.serialize(self.activation),
                'use_bias':self.use_bias,
                'kernel_initializer':tf.keras.initializers.serialize(self.kernel_initializer),
                'bias_initializer':tf.keras.initializers.serialize(self.bias_initializer),
                'kernel_regularizer':tf.keras.regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer':tf.keras.regularizers.serialize(self.bias_regularizer),
                'activity_regularizer':tf.keras.regularizers.serialize(self.activity_regularizer),
                'kernel_constraint':tf.keras.constraints.serialize(self.kernel_constraint),
                'bias_constraint':tf.keras.constraints.serialize(self.bias_constraint),
                }
        return dict(list(base_conf.items()) + list(conf_dict.items()))

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)


        if input_shape.dims[self.channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.input_dim = int(input_shape[self.channel_axis])
        kernel_shape = (self.kernel_size, self.input_dim, self.filters)

        self.kernel = self.add_weight(
                name='volume_kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)

        self.chn_mix_kernel = self.add_weight(
                name='channel_kernel',
                shape=(1, 1, 1, self.filters*3, self.filters),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
        else:
            self.bias = None

            

    def call(self, volume):
        kernel_size_d = (self.kernel_size, 1, 1, self.input_dim, self.filters)
        kernel_size_h = (1, self.kernel_size, 1, self.input_dim, self.filters)
        kernel_size_w = (1, 1, self.kernel_size, self.input_dim, self.filters)

        if self.data_format == 'channels_first':
            strides_adj =  (1, 1, self.strides, self.strides, self.strides)

            dilations_d = (1, 1, self.dilation_rate, 1, 1)
            dilations_h = (1, 1, 1, self.dilation_rate, 1)
            dilations_w = (1, 1, 1, 1, self.dilation_rate)
        else:
            strides_adj =  (1, self.strides, self.strides, self.strides, 1)

            dilations_d = (1, self.dilation_rate, 1, 1, 1)
            dilations_h = (1, 1, self.dilation_rate, 1, 1)
            dilations_w = (1, 1, 1, self.dilation_rate, 1)

        if self.data_format == 'channels_last':
            data_format='NDHWC'
        else:
            data_format='NCDHW'

        vol_conv_list = []
        for kernel_shape_adj, dilations_adj in (
                (kernel_size_d, dilations_d),
                (kernel_size_h, dilations_h),
                (kernel_size_w, dilations_w),
                ):

            kernel_adj = tf.reshape(self.kernel, kernel_shape_adj)
            vol_conv_adj = tf.nn.conv3d(
                    volume,
                    kernel_adj,
                    strides_adj,
                    padding='SAME',
                    data_format=data_format,
                    dilations=dilations_adj,
                    )

            if self.use_bias:
                if self.channel_axis == -1:
                    bias_adj = tf.reshape(self.bias, (1, 1, 1, 1, self.filters))
                else:
                    bias_adj = tf.reshape(self.bias, (1, self.filters, 1, 1, 1))

                vol_conv_adj = vol_conv_adj+bias_adj
            vol_conv_list.append(vol_conv_adj)

        vol_conv = tf.concat(vol_conv_list, self.channel_axis)
        vol_chn_mix = tf.nn.conv3d(vol_conv, self.chn_mix_kernel, (1, 1, 1, 1, 1), padding='SAME', data_format=data_format)
        return vol_chn_mix

class IsometricSeparableConv3DTranspose(tf.keras.layers.Layer):
    def __init__(self,
            filters,
            kernel_size,
            strides=1,
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if self.data_format is None:
            self.data_format = tf.keras.backend.image_data_format()
        if self.data_format == 'channels_first':
            self.channel_axis=1
        else:
            self.channel_axis=-1

        self.input_dim = None

    def get_config(self):
        base_conf = super().get_config()
        conf_dict = {
                'filters':self.filters,
                'kernel_size':self.kernel_size,
                'strides':self.strides,
                'data_format':self.data_format,
                'dilation_rate':self.dilation_rate,
                'activation':tf.keras.activations.serialize(self.activation),
                'use_bias':self.use_bias,
                'kernel_initializer':tf.keras.initializers.serialize(self.kernel_initializer),
                'bias_initializer':tf.keras.initializers.serialize(self.bias_initializer),
                'kernel_regularizer':tf.keras.regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer':tf.keras.regularizers.serialize(self.bias_regularizer),
                'activity_regularizer':tf.keras.regularizers.serialize(self.activity_regularizer),
                'kernel_constraint':tf.keras.constraints.serialize(self.kernel_constraint),
                'bias_constraint':tf.keras.constraints.serialize(self.bias_constraint),
                }
        return dict(list(base_conf.items()) + list(conf_dict.items()))

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)


        if input_shape.dims[self.channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        self.input_dim = int(input_shape[self.channel_axis])
        kernel_shape = (self.kernel_size, self.filters, self.input_dim)

        self.kernel = self.add_weight(
                name='volume_kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)

        self.chn_mix_kernel = self.add_weight(
                name='channel_kernel',
                shape=(1, 1, 1, self.filters*3, self.filters),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
        else:
            self.bias = None

            

    def call(self, volume):
        kernel_size_d = (self.kernel_size, 1, 1, self.filters, self.input_dim)
        kernel_size_h = (1, self.kernel_size, 1, self.filters, self.input_dim)
        kernel_size_w = (1, 1, self.kernel_size, self.filters, self.input_dim)

        if self.data_format == 'channels_last':
            data_format='NDHWC'
        else:
            data_format='NCDHW'

        vol_conv_list = []
        input_shape = tf.shape(volume)
        if self.data_format == 'channels_last':
            tr_output_shape = (
                    input_shape[0],
                    input_shape[1]*self.strides,
                    input_shape[2]*self.strides,
                    input_shape[3]*self.strides,
                    self.filters,
                    )
        else:
            tr_output_shape = (
                    input_shape[0],
                    self.filters,
                    input_shape[2]*self.strides,
                    input_shape[3]*self.strides,
                    input_shape[4]*self.strides,
                    )
        for kernel_shape_adj in (
                kernel_size_d,
                kernel_size_h,
                kernel_size_w,
                ):

            kernel_adj = tf.reshape(self.kernel, kernel_shape_adj)
            vol_conv_adj = tf.nn.conv3d_transpose(
                    volume,
                    kernel_adj,
                    tr_output_shape,
                    self.strides,
                    padding='SAME',
                    data_format=data_format,
                    dilations=self.dilation_rate,
                    )

            if self.use_bias:
                if self.data_format == 'channels_last':
                    bias_adj = tf.reshape(self.bias, (1, 1, 1, 1, self.filters))
                else:
                    bias_adj = tf.reshape(self.bias, (1, self.filters, 1, 1, 1))

                    vol_conv_adj = vol_conv_adj+bias_adj
            vol_conv_list.append(vol_conv_adj)

        vol_conv = tf.concat(vol_conv_list, self.channel_axis)
        vol_chn_mix = tf.nn.conv3d(vol_conv, self.chn_mix_kernel, (1, 1, 1, 1, 1), padding='SAME', data_format=data_format)
        return vol_chn_mix

