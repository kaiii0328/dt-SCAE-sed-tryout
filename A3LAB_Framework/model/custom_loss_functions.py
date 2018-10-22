from tensorflow.python.keras import backend as K
from A3LAB_Framework.utility.argument_parser import args

def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Per il margin:
    # https://www.quora.com/When-training-siamese-networks-how-does-one-determine-the-margin-for-contrastive-loss-How-do-you-convert-this-loss-to-accuracy
    """

    # https://github.com/keras-team/keras/issues/7119
    margin = args.custom_loss_margin
    # return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
