import numpy as np
import tensorflow as tf

import config as cfg
import data_features_utility
flags = tf.app.flags

flags.DEFINE_integer('train_iter', 30, 'train iteration')
flags.DEFINE_boolean('train_mode', True)
flags.DEFINE_integer('num_class', len(cfg.class_name))

flags.DEFINE_integer('mfcc_bands', cfg.mfcc_bands)
flags.DEFINE_integer('mel_spec_n_fft', cfg.mel_spec_n_fft)
flags.DEFINE_integer('angular_windowsize', cfg.angular_windowsize)
flags.DEFINE_integer('angular_n_fft', cfg.angular_n_fft)
flags.DEFINE_integer('num_TDOA', cfg.num_TDOA)

flags.DEFINE_float('dropout', 0.5)

FLAGS = flags.FLAGS

def conv_layers(input, is_training, name=None,use_bn=False):
    with tf.variable_scope('conv1' + name):
        net = tf.layers.conv2d(
            input,
            filters=64,
            kernel_size=4,
            padding='same',
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)


        pool1 = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')

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

    with tf.variable_scope('conv3' + name):
        net = tf.layers.conv2d(
            pool2,
            filters=128,
            kernel_size=5,
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)
    output = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
    output = tf.reshape(output, [-1, 2 * 2 * 128])
    return output

# mfcc(40,313) mel(128,313) angular (6,311)

def model_fn(features, labels, mode):
    mels = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])
    mfccs = tf.reshape(features['mfcc'], shape=[-1, cfg.mfcc_shape[0], cfg.mfcc_shape[1], 4])
    angulars = tf.reshape(features['angular'], shape=[-1, cfg.anguler_shape[0], cfg.anguler_shape[1], 6])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)


    mfcc_net=conv_layers(mfccs,is_training,name='mfcc')
    mel_net=conv_layers(mels,is_training,name='mel')
    angular_net=conv_layers(angulars,is_training,name='angular')

    gru1_in = tf.concat([mfcc_net, mel_net, angular_net], 0)

    gru1_layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(256,kernel_initializer=tf.orthogonal_initializer)] * 2)
    gru1_out, gru1_state = tf.nn.dynamic_rnn(gru1_layer, gru1_in, dtype=tf.float32, scope='gru1')

    gru2_Layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(256,kernel_initializer=tf.orthogonal_initializer) * 2])
    gru2_out, gru2_state = tf.nn.dynamic_rnn(gru2_Layer, gru1_out, dtype=tf.float32, scope='gru2')

    # logits=tf.transpose(gru2_out,[1,0,2])
    # logits=tf.gather(logits)
    # dense1=tf.layers.dense(gru2_out)
    # dropouts=tf.nn.dropout(logits)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./CRNN")

    train_input_fn=data_features_utility.AudioPrepare.tf_input_fn(is_training=False)
    # train_input_fn=tf.estimator.inputs.numpy_input_fn(
    #     x={'x':xx},
    #     y=,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True
    # )
    classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=None
    )
    pass


if __name__ == '__main__':
    tf.app.run()
