
# -*- coding: gbk -*-


from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

from keras.utils import multi_gpu_model
import keras.backend as K
from keras.constraints import max_norm
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop,Adam
#import keras.backend.tensorflow_backend as K
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.metrics import  precision_score ,recall_score
import heapq
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K1
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from sklearn.metrics import f1_score
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation

from keras.regularizers import l2
from keras import initializers, layers



from keras import activations
from keras import backend as K
from keras.engine.topology import Layer

#kernel_regularizer=l2(0.0002),recurrent_regularizer=l2(0.0002),

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

class Classifier(object):
    def __init__(self,
             weights,
             max_feature_value,
             max_sequence_len,
             num_classes,
             embedding_dim = 200):
        self.weights = weights
        self.max_feature_value = max_feature_value
        self.max_sequence_len = max_sequence_len
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.input_shape = (self.max_sequence_len,)

        #self.weights = {0 : 0.025, 1 : 0.975}
        
    def __call__(self):
        input_tensor = Input(shape = self.input_shape)
        return self.build_model(input_tensor)
    def build_model(self, x):
    
        input_tensor = x

        em = Embedding(input_dim=self.max_feature_value,
                                 output_dim=self.embedding_dim,
                                 mask_zero=False,
                                 weights=[self.weights],
                                 input_length=self.max_sequence_len,
                                 trainable=False,
                                 name = "Embedding")(x)
        x = Dropout(0.5)(em)
        #x = Bidirectional(CuDNNLSTM(256,kernel_regularizer=l2(0.0002),recurrent_regularizer=l2(0.0002),return_sequences=True))(x)
        """
        x = Bidirectional(LSTM(units = 256,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   bias_initializer='zeros',
                   unit_forget_bias=True,
                   kernel_regularizer=None,
                   recurrent_regularizer=None, 
                   bias_regularizer=None, 
                   activity_regularizer=None, 
                   kernel_constraint=None,
                   recurrent_constraint=None,
                   bias_constraint=None, 
                   return_sequences = True,
                   dropout=0.5,
                   recurrent_dropout=0.5,
                   name = "Lstm"))(x)
        """
        
        #x = Concatenate()([x,em])
       
        
        x1 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=32, kernel_size=1, padding="same")(x)))
        x1 = GlobalMaxPooling1D()(x1)
        x2 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=32, kernel_size=2, padding="same")(x)))
        x2 = GlobalMaxPooling1D()(x2)
        x3 = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=32, kernel_size=3, padding="same")(x)))
        x3 = GlobalMaxPooling1D()(x3)
        
                

        x = Concatenate()([x1,x2,x3])
        
        x = Dropout(0.5)(x)     
        
        """
        x = PrimaryCap(x, dim_vector=32, n_channels=4, kernel_size=3, strides=1, padding='valid')

        x = BatchNormalization()(x)
        x = CapsuleLayer(num_capsule=self.num_classes, dim_capsule=16, routings=3, name='digitcaps')(x)
        out_caps = Length(name='out_caps')(x)        
        output_tensor = out_caps
        """
                
        
        x = Dense(self.num_classes, activation = 'sigmoid')(x)
        output_tensor = x
        
        model = Model([input_tensor], [output_tensor])

        
        #mulmodel = multi_gpu_model(model, 2)
        
        """
        model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,
              optimizer=Adam(0.001),
              metrics=['accuracy'])
        """
        model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
        
        return model

      
    

    
