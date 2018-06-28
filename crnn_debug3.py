import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
import dataset  as data_utility
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model(input, is_training, name):
    # Convolutional Layer #1
    with tf.variable_scope('conv1' + name):
        net = tf.layers.conv2d(
            inputs=input,
            filters=16,
            kernel_size=[3, 3],
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
            filters=32,
            kernel_size=[3, 3],
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
            filters=64,
            kernel_size=[3, 3],
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
    return net


def model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], shape=[-1, 28, 28, 1])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    net = cnn_model(input_layer, is_training, '')

    with tf.variable_scope('Reshape_cnn'):
        output_shape = net.get_shape().as_list()  # [batch,height,width,features]
        net = tf.transpose(net, [0, 2, 1, 3])
        net = tf.reshape(net, shape=[-1, output_shape[2], output_shape[1] * output_shape[3]])

    with tf.variable_scope('bi_GRU'):
        fw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                                          state_keep_prob=0.5) for _ in range(3)]
        bw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
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

        for var in tf.global_variables():
            scope = var.name.split('/')[0]
            if scope == 'bi_GRU':
                gates_candidate = var.name.split('/')[-2]
                fw_cell = var.name.split('/')[2] + '_' + var.name.split('/')[-4] + '_'
                kernal_bise = var.name.split('/')[-1].split(':')[0]
                if gates_candidate == 'candidate':
                    tf.summary.histogram('bi_GRU_' + fw_cell + kernal_bise, var)

        # rnn_outputs_merged = tf.concat(rnn_outputs, 2)
        # rnn_finial = tf.unstack(rnn_outputs_merged, rnn_outputs_merged.get_shape().as_list()[1], 1)[-1]

    with tf.variable_scope('dense_layer'):
        # logits = tf.layers.dense(inputs=rnn_finial, units=10, activation=tf.nn.relu)
        net = tf.layers.dense(inputs=last_state_fw[-1] + last_state_bw[-1], units=512, activation=tf.nn.relu)
        net = tf.layers.dropout(inputs=net, rate=0.4, training=is_training)
        logits = tf.layers.dense(inputs=net, units=10, activation=tf.nn.relu)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1))
    accuracy = tf.Print(accuracy, [accuracy], 'Acuracy__')
    tf.summary.scalar('train_accuracy', accuracy[1])

    predictions = {
        'classes': tf.argmax(tf.nn.softmax(logits), axis=1, name='predict_class'),
        'prob': tf.nn.softmax(logits, name='softmax_tensor'),
        'training_accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1),
                                                 name='xxx'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
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
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mnist = tf.contrib.learn.datasets.DATASETS['mnist']('./tmp/mnist')
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./crnn_model_debug")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
