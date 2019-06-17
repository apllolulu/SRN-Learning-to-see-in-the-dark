import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from util.BasicConvLSTMCell import *

cardinality = 8 # how many split ?
blocks = 5 # res_block ! (split + transition)
depth = 64 # out channel
reduction_ratio = 4



def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)



class SE_ResNeXt():
    def __init__(self, x, cell,rnn_state,training,scope='se_resnet'):
        self.training = training
        #self.cell =cell
        #self.rnn_state = rnn_state
        self.encoder_decoder = self.Build_SEnet_encoder_decoder(x,cell,rnn_state,scope='se_resnet')


    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=32, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :


            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                if input_dim // 2 == out_dim:
                    #x = conv_layer(input_x, filter=out_dim, kernel=[3, 3], stride=1, layer_name='decoder_conv'+layer_num+'_'+str(i))
                    #x = Batch_Normalization(x, training=self.training, scope='se'+'_batch_'+str(i))
                    #input_x = self.transition_layer(input_x, out_dim=out_dim, scope='input_layer_' + layer_num + '_' + str(i))
                    #input_x = tf.layers.conv2d(inputs=input_x, use_bias=False, filters=out_dim, kernel_size=[3, 3],
                    #                          strides=1, padding='SAME')
                    input_x = conv_layer(input_x, filter=out_dim, kernel=[1, 1], stride=1, layer_name='input_layer_' + 'conv_'+str(i))
                    input_x = Relu(input_x)
                    pad_input_x = input_x

                else:
                    pad_input_x = input_x

            input_x = Relu(x + pad_input_x)
        #print("input_x.shape",input_x.shape)
        return input_x


    def Build_SEnet_encoder_decoder(self, input_x,cell,rnn_state,scope='se_resnet'):
        with tf.variable_scope(scope):
            #print("1:", input_x.shape)  # (16, 64, 64, 2)
            input_x = self.first_layer(input_x, scope="first_layer")
            #print("2:", input_x.shape)
            #x = self.residual_layer(input_x, out_dim=32, layer_num='encoder_1')
            #print("3:", x.shape)
            x = self.residual_layer(input_x, out_dim=64, layer_num='encoder_2')
            #print("4:", x.shape)
            #x = self.residual_layer(x, out_dim=64, layer_num='encoder_3')
            #print("5:", x.shape)
            x = self.residual_layer(x, out_dim=128, layer_num='encoder_4')
            #print("6:", x.shape)
            #x = self.residual_layer(x, out_dim=128, layer_num='encoder_5')
            #print("7:", x.shape) # 7: (16, 16, 16, 128)

            x, rnn_state = cell(x, rnn_state)

            x = self.residual_layer(x, out_dim=128, layer_num='decoder_1')
            #x = self.residual_layer(x, out_dim=128, layer_num='decoder_2')
            print("decoder_2:", x.shape)  # 16 16 16 128

            #x = self.residual_layer(x, out_dim=64, layer_num='decoder_3')
            x = self.residual_layer(x, out_dim=64, layer_num='decoder_4')
            print("decoder_4:", x.shape)

            #x = self.residual_layer(x, out_dim=32, layer_num='decoder_5')
            x = self.residual_layer(x, out_dim=32, layer_num='decoder_6')
            print("decoder_6:", x.shape)

            return x,rnn_state







