import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.losses import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

# def dice_coef_binary(y_true, y_pred, smooth=1e-7):
#     '''
#     Dice coefficient for 2 categories. Ignores background pixel label 0
#     Pass to model as metric during compile statement
#     '''
#     # y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2)[...,1:])
#     # y_pred_f = K.flatten(y_pred[...,1:])
#     intersect = tf.reduce_sum(y_true_f * y_pred_f, axis=(1,2,3))
#     denom = tf.reduce_sum(y_true_f + y_pred_f, axis=(1,2,3))
#     return tf.reduce_mean((2. * intersect / (denom + smooth)))


# def dice_loss(y_true, y_pred):
#     '''
#     Dice loss to minimize. Pass to model as loss during compile statement
#     '''
#     return 1 - dice_coef_binary(y_true, y_pred)


# def binary_crossentropy(target, output, from_logits=False):
#     """Binary crossentropy between an output tensor and a target tensor.

#     # Arguments
#         target: A tensor with the same shape as `output`.
#         output: A tensor.
#         from_logits: Whether `output` is expected to be a logits tensor.
#             By default, we consider that `output`
#             encodes a probability distribution.

#     # Returns
#         A tensor.
#     """
#     # Note: tf.nn.sigmoid_cross_entropy_with_logits
#     # expects logits, Keras expects probabilities.
#     if not from_logits:
#         # transform back to logits
#         _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
#         output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
#         output = tf.log(output / (1 - output))

#     return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    y_pred = tf.nn.sigmoid(y_pred)
    numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
    denominator = tf.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    
    return 1 - tf.reduce_sum(numerator / (denominator + epsilon)) # average over classes and batch



def dice_loss(y_true, y_pred):
    # y_pred = tf.nn.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

def comb_loss(y_true, y_pred):
    return soft_dice_loss(y_true, y_pred)# + binary_crossentropy(y_pred,y_true)


# def dice_loss(y_true, y_pred):
#   numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
#   denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

#   return 1 - numerator / denominator

def dmnet(pretrained_weights=None, input_size=(512, 512, 1)):

    ###### Albu Path ###################
    inputA = Input(input_size)
    z = Conv2D(16, (3, 3), activation='relu', padding='same')(inputA)
    z = Dropout(0.5)(z)
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(32, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)
    pathA = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(z)

    ####### DM Path #####
    inputDM = Input(input_size)

    z = Conv2D(16, (3, 3), activation='relu', padding='same')(inputDM)
    z = Dropout(0.5)(z)
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(32, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)


    pathDM = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(z)
    
    #### Merge Path ####
    merge1 = Concatenate(axis = -1)([pathA, pathDM])


    z = Conv2D(256, (3, 3), activation='relu', padding='same')(merge1)
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)
    z = Dropout(0.25)(z)
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(32, (3, 3), activation='relu', padding='same')(z)
    z = Conv2D(16, (3, 3), activation='relu', padding='same')(z)


    op = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(z)
    print(op)

    model = Model(inputs=[inputA, inputDM], output=op)
    adam = Adam(lr=0.0001, decay=1e-5)
    adagrad = Adagrad(lr=0.001,decay=1e-4)
    adadelta = Adadelta(lr=0.001,decay=1e-4)
    sgd = SGD(lr=0.001,decay=1e-4)
    model.compile(optimizer=adam, loss=comb_loss)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
