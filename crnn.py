import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
# import dataset  as data_utility
import data_features_utility as data_utility
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model(input, is_training, name):
    # Convolutional Layer #1
    with tf.variable_scope('conv1' + name):
        net = tf.layers.conv2d(
            inputs=input,
            filters=8,
            kernel_size=[5, 5],
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        weight = [var for var in tf.global_variables() if var.name == 'conv1/conv2d/kernel:0'][0]
        tf.summary.histogram('conv_kernel', weight)
        bias = [var for var in tf.global_variables() if var.name == 'conv1/conv2d/bias:0'][0]
        tf.summary.histogram('conv_bise', bias)
        bn_gamma = [var for var in tf.global_variables() if var.name == 'conv1/batch_normalization/gamma:0'][0]
        tf.summary.histogram('bn_gamma', bn_gamma)
        bn_beta = [var for var in tf.global_variables() if var.name == 'conv1/batch_normalization/beta:0'][0]
        tf.summary.histogram('bn_beta', bn_beta)

    with tf.variable_scope('conv2' + name):
        net = tf.layers.conv2d(
            inputs=net,
            filters=16,
            kernel_size=[5, 5],
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        weight = [var for var in tf.global_variables() if var.name == 'conv2/conv2d/kernel:0'][0]
        tf.summary.histogram('conv_kernel', weight)
        bias = [var for var in tf.global_variables() if var.name == 'conv2/conv2d/bias:0'][0]
        tf.summary.histogram('conv_bise', bias)
        bn_gamma = [var for var in tf.global_variables() if var.name == 'conv2/batch_normalization/gamma:0'][0]
        tf.summary.histogram('bn_gamma', bn_gamma)
        bn_beta = [var for var in tf.global_variables() if var.name == 'conv2/batch_normalization/beta:0'][0]
        tf.summary.histogram('bn_beta', bn_beta)

    with tf.variable_scope('conv3' + name):
        net = tf.layers.conv2d(
            inputs=net,
            filters=32,
            kernel_size=[5, 5],
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        # time_axis_pooling_shape=net.get_shape().as_list()[2]
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=[1,time_axis_pooling_shape], strides=2)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        weight = [var for var in tf.global_variables() if var.name == 'conv3/conv2d/kernel:0'][0]
        tf.summary.histogram('conv_kernel', weight)
        bias = [var for var in tf.global_variables() if var.name == 'conv3/conv2d/bias:0'][0]
        tf.summary.histogram('conv_bise', bias)
        bn_gamma = [var for var in tf.global_variables() if var.name == 'conv3/batch_normalization/gamma:0'][0]
        tf.summary.histogram('bn_gamma', bn_gamma)
        bn_beta = [var for var in tf.global_variables() if var.name == 'conv3/batch_normalization/beta:0'][0]
        tf.summary.histogram('bn_beta', bn_beta)

    with tf.variable_scope('conv4'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[1, 39], strides=[1, 1])
        weight = [var for var in tf.global_variables() if var.name == 'conv4/conv2d/kernel:0'][0]
        tf.summary.histogram('conv_kernel', weight)
        bias = [var for var in tf.global_variables() if var.name == 'conv4/conv2d/bias:0'][0]
        tf.summary.histogram('conv_bise', bias)
        bn_gamma = [var for var in tf.global_variables() if var.name == 'conv4/batch_normalization/gamma:0'][0]
        tf.summary.histogram('bn_gamma', bn_gamma)
        bn_beta = [var for var in tf.global_variables() if var.name == 'conv4/batch_normalization/beta:0'][0]
        tf.summary.histogram('bn_beta', bn_beta)

    return net


def model_fn(features, labels, mode):
    input_layer = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = cnn_model(input_layer, is_training, '')

    with tf.variable_scope('Reshape_cnn'):
        output_shape = net.get_shape().as_list()  # [batch,height,width,features]
        net = tf.transpose(net, [0, 2, 1, 3])
        net = tf.reshape(net, shape=[-1, output_shape[2], output_shape[1] * output_shape[3]])

    with tf.variable_scope('bi_GRU'):
        fw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(1024, kernel_initializer=tf.orthogonal_initializer),
                                          state_keep_prob=0.5) for _ in range(3)]
        bw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(1024, kernel_initializer=tf.orthogonal_initializer),
                                          state_keep_prob=0.5) for _ in range(3)]

        # fw_cell_list = [
        #     tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
        #                                   input_keep_prob=0.8, output_keep_prob=0.8) for _ in range(3)]
        # bw_cell_list = [
        #     tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
        #                                   input_keep_prob=0.8, output_keep_prob=0.8) for _ in range(3)]

        multi_rnn_fW_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
        multi_rnn_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)

        rnn_outputs, (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_rnn_fW_cell,
            cell_bw=multi_rnn_bw_cell,
            inputs=net,
            dtype=tf.float32)

        # rnn_outputs_merged = tf.concat(rnn_outputs, 2)
        # rnn_finial = tf.unstack(rnn_outputs_merged, rnn_outputs_merged.get_shape().as_list()[1], 1)[-1]

        # record rnn cells
        for var in tf.global_variables():
            scope = var.name.split('/')[0]
            if scope == 'bi_GRU':
                gates_candidate = var.name.split('/')[-2]
                fw_cell = var.name.split('/')[2] + '_' + var.name.split('/')[-4] + '_'
                kernal_bise = var.name.split('/')[-1].split(':')[0]
                if gates_candidate == 'candidate':
                    tf.summary.histogram('bi_GRU_' + fw_cell + kernal_bise, var)

    with tf.variable_scope('dense_layer'):
        rnn_outputs_merged = tf.concat(rnn_outputs, 2)
        rnn_finial = tf.squeeze(rnn_outputs_merged, 1)

        weight = tf.get_variable('birnn_out_weight', [2 * 1024, 1024], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        bise = tf.get_variable('birnn_out_bise', [1024], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        net = tf.matmul(rnn_finial, weight) + bise

        # net = tf.layers.dense(inputs=rnn_finial, units=2048, activation=tf.nn.relu)
        # net = tf.layers.dense(inputs=last_state_fw[-1] + last_state_bw[-1], units=512, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.4, training=is_training)
        logits = tf.layers.dense(inputs=net, units=cfg.num_class, activation=tf.nn.relu)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1))
    accuracy = tf.Print(accuracy, [accuracy], 'Acuracy__')
    tf.summary.scalar('train_accuracy', accuracy[1])

    predictions = {
        'classes': tf.argmax(tf.nn.softmax(logits), axis=1, name='predict_class'),
        'prob': tf.nn.softmax(logits, name='softmax_tensor'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric)


def main(unused_argv):
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./crnn_model_birnn_test")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    data_solution = data_utility.AudioPrepare()
    train_input_fn = data_solution.tf_input_fn_maker(is_training=True, n_epoch=100)

    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    test_solution = data_utility.AudioPrepare()
    test_input_fn = test_solution.tf_input_fn_maker(is_training=False, n_epoch=1)

    # eval_results = classifier.evaluate(input_fn=test_input_fn, steps=100)
    # print(eval_results)
    eval_results = classifier.evaluate(input_fn=test_input_fn, steps=3000)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
