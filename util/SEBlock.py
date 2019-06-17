import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

import numpy as np



def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)


def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)


# 引入 Squeeze_excitation_block
def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):

    squeeze = Global_Average_Pooling(input_x)

    excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
    excitation = Relu(excitation)
    excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
    excitation = Sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

    scale = input_x * excitation
    print("SE BLOCK SUCCEED")
    return scale