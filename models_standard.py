from keras.layers import *
from keras.models import Model
import tensorflow as tf

def var_network(var, hidden=10, output=4):
    var = Flatten()(var)
    var = Dense(hidden, activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01))(var)
    var = Dense(hidden, activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01))(var)
    return Dense(output, kernel_regularizer=tf.keras.regularizers.L1L2(0.01))(var)

def conv_network(var, n_filters=5, kernel_size=3):
    var = SeparableConv2D(n_filters, kernel_size, activation='tanh',
                          depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
                          pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
                          activity_regularizer=tf.keras.regularizers.L2(0.01))(var)
    var = Conv2D(n_filters, 1, activation='tanh',
                 kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                 activity_regularizer=tf.keras.regularizers.L2(0.01))(var)
    return var

def CreateModel(shape, n_filters, pool_size):
    x_in = Input(shape)
    stack = conv_network(x_in, n_filters=n_filters)
    stack = AveragePooling2D(pool_size=(pool_size, pool_size), padding="valid")(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model