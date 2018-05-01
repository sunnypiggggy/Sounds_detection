import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config as cfg


def conv_layers(input, is_training, name=None,use_bn=False):
    with tf.variable_scope('conv1' + name):
        net = tf.layers.conv2d(
            input,
            filters=128,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)


        pool1 = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
        pool1 = tf.layers.dropout(pool1, rate=0.5, training=is_training)

    with tf.variable_scope('conv2' + name):
        net = tf.layers.conv2d(
            pool1,
            filters=128,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)

        pool2 = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
        pool2 = tf.layers.dropout(pool2,rate=0.5,training=is_training)


    with tf.variable_scope('conv3' + name):
        net = tf.layers.conv2d(
            pool2,
            filters=128,
            kernel_size=5,
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)

        pool3 = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')

    conv_output=tf.layers.dropout(pool3,rate=0.5,training=is_training)
    with tf.variable_scope('Reshape_cnn'+name):
        output_shape=conv_output.get_shape().as_list()#[batch,height,width,features]
        output=tf.transpose(conv_output,[0,2,1,3],name='transposed')
        output=tf.reshape(output,shape=[-1,output_shape[2],output_shape[1]*output_shape[3]],name='reshaped')#[batch,width,heigth*features]
    return output


def model_fn(features, labels, mode):
    mels = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])
    mfccs = tf.reshape(features['mfcc'], shape=[-1, cfg.mfcc_shape[0], cfg.mfcc_shape[1], 4])
    angulars = tf.reshape(features['angular'], shape=[-1, cfg.anguler_shape[0], cfg.anguler_shape[1], 6])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)


    mfcc_net=conv_layers(mfccs,is_training,name='mfcc')
    mel_net=conv_layers(mels,is_training,name='mel')
    angular_net=conv_layers(angulars,is_training,name='angular')

    gru1_in = tf.concat([mfcc_net, mel_net, angular_net], 1)

    gru1_layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(512,kernel_initializer=tf.orthogonal_initializer)] * 2)
    gru1_out, gru1_state = tf.nn.dynamic_rnn(gru1_layer, gru1_in, dtype=tf.float32, scope='gru1')

    gru2_Layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(512,kernel_initializer=tf.orthogonal_initializer) * 2])
    gru2_out, gru2_state = tf.nn.dynamic_rnn(gru2_Layer, gru1_out, dtype=tf.float32, scope='gru2')

if __name__ =='__main__':

    # x=tf.placeholder(tf.float32,shape=(None,10,1024))
    #
    # gru1_layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(1024,kernel_initializer=tf.orthogonal_initializer)] * 2)
    # gru1_out, gru1_state = tf.nn.dynamic_rnn(gru1_layer, x, dtype=tf.float32, scope='gru1')

    is_training=True

    mfccs=tf.placeholder(tf.float32,shape=[None,40,313,4])
    mels = tf.placeholder(tf.float32, shape=[None, 128, 313, 4])
    angulars = tf.placeholder(tf.float32, shape=[None, 80, 313, 6])

    mfcc_net=conv_layers(mfccs,is_training,name='mfcc')
    mel_net=conv_layers(mels,is_training,name='mel')
    angular_net=conv_layers(angulars,is_training,name='angular')

    # max_axis1=max(mfcc_net.getshape[1],mel_net.shape)
    # angular_net=tf.pad(angular_net,[[0,0,],[0,1,0,]],"CONSTANT")
    gru1_in = tf.concat([mfcc_net, mel_net, angular_net], axis=2)

    gru1_layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(6400,kernel_initializer=tf.orthogonal_initializer)] * 2)
    gru1_out, gru1_state = tf.nn.dynamic_rnn(gru1_layer, gru1_in, dtype=tf.float32, scope='gru1')

    gru2_Layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(384,kernel_initializer=tf.orthogonal_initializer) * 2])
    gru2_out, gru2_state = tf.nn.dynamic_rnn(gru2_Layer, gru1_out, dtype=tf.float32, scope='gru2')


    config=tf.ConfigProto(device_count={'GPU':0},log_device_placement=True)

    with tf.Session(config=config) as sess:
        hello=tf.constant('shit happens')
        print(sess.run(hello))








