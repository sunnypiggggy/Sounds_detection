import tensorflow as tf
import numpy as np
import scipy as sci

def conv_layers(input, is_training, name=None):
    with tf.variable_scope('conv1' + name):
        conv1 = tf.layers.conv2d(
            input,
            filters=64,
            kernel_size=4,
            padding='same',
            activation=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(conv1, training=is_training)

    pool1 = tf.layers.max_pooling2d(bn1, pool_size=2, strides=2, padding='same')
    # pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')

    with tf.variable_scope('conv2' + name):
        conv2 = tf.layers.conv2d(
            pool1,
            filters=128,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu)
        bn2 = tf.layers.batch_normalization(conv2, training=is_training)

    pool2 = tf.layers.max_pooling2d(bn2, pool_size=2, strides=2, padding='same')
    # pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='same')
    with tf.variable_scope('conv3' + name):
        conv3 = tf.layers.conv2d(
            pool2,
            filters=128,
            kernel_size=5,
            activation=tf.nn.relu)
        bn3 = tf.layers.batch_normalization(conv3, training=is_training)
    output = tf.layers.max_pooling2d(bn3, pool_size=2, strides=2, padding='same')
    # output = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='same')
    output = tf.reshape(output, [-1, 2 * 2 * 128])
    return output


def crnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv_out = conv_layers(input_layer, (mode == tf.estimator.ModeKeys.TRAIN), name='test_conv')

    dense = tf.layers.dense(inputs=conv_out, units=512, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "class": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits=logits, name='softmax_tensor'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions['class'])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    mnist = tf.contrib.learn.datasets.DATASETS['mnist']('./tmp/mnist')
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=crnn_model_fn, model_dir="./tmp/mnist_convnet_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=2000,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
