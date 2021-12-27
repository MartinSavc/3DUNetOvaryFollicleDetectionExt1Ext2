from functools import partial
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

def dice_coeff(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return -dice_coeff(y_true, y_pred)

def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                                axis=axis) + smooth))

def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coeff(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def rho_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = K.reshape(y_true, [-1])
    y_pred_f = K.reshape(y_pred, [-1])
    intersection = K.sum(y_true_f * y_pred_f)
    score = ((intersection + smooth) * (intersection + smooth)) / ((K.sum(y_true_f)+smooth) * (K.sum(y_pred_f) + smooth))
    return score

def rho_loss(y_true, y_pred):
    loss = -rho_coeff(y_true, y_pred)
    return loss

def acc_coeff(y_true, y_pred):
    y_true_f = K.reshape(y_true, [-1])
    y_pred_f = K.reshape(y_pred, [-1])    
    TP = K.sum(y_true_f * y_pred_f)    
    FN = K.sum(y_true_f) - TP
    FP = K.sum(y_pred_f) - TP
    
    U = K.cast(K.prod(K.shape(y_true_f)), 'float32')
    TN = U - TP - FN - FP

    score = (TP + TN) / U        
    return score

def acc_loss(y_true, y_pred):
    loss = -acc_coeff(y_true, y_pred)
    return loss

# dice_coef = dice_coefficient
# dice_coef_loss = dice_coefficient_loss


def bce_dice_rho_loss(y_true, y_pred):        
#    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + rho_loss(y_true, y_pred)
    # loss = dice_loss(y_true, y_pred) + rho_loss(y_true, y_pred) + acc_loss(y_true, y_pred)
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + rho_loss(y_true, y_pred) + acc_loss(y_true, y_pred)
    return loss
