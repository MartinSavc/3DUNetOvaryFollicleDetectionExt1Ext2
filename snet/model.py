import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras

def snet_model():
    input_chns=3
    output_chns=6
    depth=4
    chns_base=16
    kernel_size=3
    dropout_rate=0.5

    tens_input=keras.Input((None, None, input_chns))


    tens=tens_input
    encoded_tens_list=[]
    for l in range(depth):
        chns=chns_base*(2**l)

        if l==depth-1:
            tens=keras.layers.Dropout(dropout_rate)(tens)

        for n in range(2):
            tens=keras.layers.Conv2D(chns, kernel_size, padding='same')(tens)
            tens=keras.layers.BatchNormalization()(tens)
            tens=keras.layers.Activation('relu')(tens)


        if l<depth-1:
            encoded_tens_list.append(tens)
            tens=keras.layers.MaxPool2D()(tens)

    tens_upsampled=keras.layers.UpSampling2D(2**(depth-1))(tens)
    output_tens=keras.layers.Conv2D(output_chns, 1, padding='same', activation='sigmoid')(tens_upsampled)
    outputs_list=[output_tens]

    for l in range(depth-1, 0, -1):
        encoded_tens = encoded_tens_list.pop()
        chns=chns_base*(2**(l-1))
        tens=keras.layers.Conv2DTranspose(chns, 2, strides=2, padding='same')(tens)
        tens=keras.layers.Add()([tens, encoded_tens])
        for n in range(2):
            tens=keras.layers.Conv2D(chns, kernel_size, padding='same')(tens)
            tens=keras.layers.BatchNormalization()(tens)
            tens=keras.layers.Activation('relu')(tens)

        if l > 1:
            tens_upsampled=keras.layers.UpSampling2D(2**(l-1))(tens)
        else:
            tens_upsampled = tens
        output_tens=keras.layers.Conv2D(output_chns, 1, padding='same', activation='sigmoid')(tens_upsampled)
        outputs_list.append(output_tens)

        model = keras.Model(inputs=tens_input, outputs=outputs_list)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=composite_bce_loss)

    return model


def weighted_bce_loss(y_true, y_pred, alpha, beta):
    y_true = K.reshape(y_true, [-1])
    y_pred = K.reshape(y_pred, [-1])

    return -1*K.sum(alpha*y_true*K.log(y_pred+K.epsilon()) + beta*(1-y_true)*K.log(1-y_pred+K.epsilon()))

def composite_bce_loss(y_true, y_pred):
    y_f_true = y_true[:,:, 1::2]
    y_o_true = y_true[:,:, 0::2]
    y_f_pred = y_pred[:,:, 1::2]
    y_o_pred = y_pred[:,:, 0::2]
    return weighted_bce_loss(y_o_true, y_o_pred, 1, 1)+\
           weighted_bce_loss(y_f_true, y_f_pred, 3, 1)+\
           weighted_bce_loss(y_o_true, y_f_pred, 1, 1)




        

