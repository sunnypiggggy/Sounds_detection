import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
import dataset  as data_utility
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)

import module.capsuleNetwork as cap


def model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28,1])
    with tf.variable_scope('cap_network'):
        capLayer = cap.CapsuleLayer(2, 3, 10, layer_type='conv', vars_scope='conv_cap')

        net = capLayer(input_layer, [3, 3],(1,1,1,1), 3)
        capLayerDNN = cap.CapsuleLayer(2, 3, 10, layer_type='dnn', vars_scope='dnn_cap')
        # net = capLayerDNN(input_layer, [3, 3], (1,2,2,1), 3)
        net = capLayerDNN(input_layer, [3, 3], (1,1,1,1), 3)


    logits=tf.layers.dense(inputs=net,units=10,activation=tf.nn.relu)

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

if __name__ == '__main__':
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mnist = tf.contrib.learn.datasets.DATASETS['mnist']('./tmp/mnist')
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./capsule_model")

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

    pass
