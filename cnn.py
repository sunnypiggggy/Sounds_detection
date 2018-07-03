import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
import dataset  as data_utility
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    # input_layer = tf.reshape(features["mel"], [-1, 128, 313, 4])
    input_layer = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])
    # input_layer = tf.reshape(features['angular'], shape=[-1, cfg.anguler_shape[0], cfg.anguler_shape[1], 6])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # input_layer=features['mel'].set_shape([ cfg.mel_shape[0], cfg.mel_shape[1], 4])
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]'
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    with tf.variable_scope('conv1'):
        net = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=5,
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net,  training=is_training)
        net=tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net=tf.layers.dropout(net,0.5,training=is_training)

    with tf.variable_scope('conv2'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=5,
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.variable_scope('conv3'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=5,
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)


    with tf.variable_scope('conv4'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=5,
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net,  training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)



    with tf.variable_scope('conv5'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=5,
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)


    with tf.variable_scope('conv6'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=5,
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net,  training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)


    # net = tf.reshape(net, [-1,2*4*64])
    net=tf.layers.flatten(net)
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    # dense = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=net, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=9,activation=tf.nn.relu)


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),

        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            # optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3,momentum=0.9)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            optimize = optimizer.apply_gradients(zip(gradients, variables))
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1))
            tf.summary.scalar('train_accuracy', accuracy[1])
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./cnn_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    hook = tf_debug.TensorBoardDebugHook("sunny-workstation:7000")

    test_solution = data_utility.AudioPrepare()
    train_input_fn = test_solution.tf_input_fn_maker(is_training=False, n_epoch=100)

    # Evaluate the model and print results
    test_solution = data_utility.AudioPrepare()
    test_input_fn = test_solution.tf_input_fn_maker(is_training=False, n_epoch=10)

    for _ in range(100):
        classifier.train(
            input_fn=train_input_fn,
            steps=1000,
            hooks=[logging_hook])



        eval_results = classifier.evaluate(input_fn=test_input_fn, steps=300)
        print(eval_results)

if __name__ == "__main__":
    tf.app.run()
