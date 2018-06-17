import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
import data_features_utility as data_utility
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)


def rnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28])
    with tf.variable_scope('bi_GRU'):
        # fw_cell_list = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256,kernel_initializer=tf.orthogonal_initializer),state_keep_prob=0.5) for _ in range(3)]
        # bw_cell_list = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256,kernel_initializer=tf.orthogonal_initializer),state_keep_prob=0.5) for _ in range(3)]

        fw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                                          input_keep_prob=0.8, output_keep_prob=0.8) for _ in range(3)]
        bw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                                          input_keep_prob=0.8, output_keep_prob=0.8) for _ in range(3)]

        multi_rnn_fW_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
        multi_rnn_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)

        rnn_outputs, (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_rnn_fW_cell,
            cell_bw=multi_rnn_bw_cell,
            inputs=input_layer,
            dtype=tf.float32)

        rnn_outputs_merged = tf.concat(rnn_outputs, 2)
        rnn_finial = tf.unstack(rnn_outputs_merged, rnn_outputs_merged.get_shape().as_list()[1], 1)[-1]
    # fc_out = tf.layers.dense(inputs=rnn_finial, units=512, activation=tf.nn.relu)
    # logits = tf.layers.dense(inputs=fc_out, units=10,activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=rnn_finial, units=10, activation=tf.nn.relu)

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
        model_fn=rnn_model_fn, model_dir="./rnn_model")

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
