import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
import data_features_utility as data_utility


def weightVar(shape, mean=0.0, stddev=0.02, name='weights'):
    init_w = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(init_w, name=name)


def biasVar(shape, value=0.0, name='bias'):
    init_b = tf.constant(value=value, shape=shape)
    return tf.Variable(init_b, name=name)


def conv_layers(input, is_training, pooling_config=None, name=None, use_bn=False):
    if pooling_config == None:
        pooling_config = [2, 2, 2]
    with tf.variable_scope('conv1' + name):

        net = tf.layers.conv2d(
            input,
            filters=128,
            kernel_size=5,
            padding='same',
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)

        pool1 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[0], 2), padding='same')
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

        pool2 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[1], 2), padding='same')
        pool2 = tf.layers.dropout(pool2, rate=0.5, training=is_training)

    with tf.variable_scope('conv3' + name):
        net = tf.layers.conv2d(
            pool2,
            filters=128,
            kernel_size=5,
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)

        pool3 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[2], 2), padding='same')

        conv_output = tf.layers.dropout(pool3, rate=0.5, training=is_training)
    with tf.variable_scope('Reshape_cnn' + name):
        output_shape = conv_output.get_shape().as_list()  # [batch,height,width,features]
        output = tf.transpose(conv_output, [0, 2, 1, 3], name='transposed')
        output = tf.reshape(output, shape=[-1, output_shape[2], output_shape[1] * output_shape[3]],
                            name='reshaped')  # [batch,width,heigth*features]
    return output


def model_fn(features, labels, mode):
    # mels = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])
    # mfccs = tf.reshape(features['mfcc'], shape=[-1, cfg.mfcc_shape[0], cfg.mfcc_shape[1], 4])
    # angulars = tf.reshape(features['angular'], shape=[-1, cfg.anguler_shape[0], cfg.anguler_shape[1], 6])
    mels = features['mel']
    mfccs = features['mfcc']
    angulars = features['angular']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    mfcc_net = conv_layers(mfccs, is_training, pooling_config=[2, 2, 2], name='mfcc')
    mel_net = conv_layers(mels, is_training, pooling_config=[4, 2, 2], name='mel')
    angular_net = conv_layers(angulars, is_training, pooling_config=[4, 2, 2], name='angular')

    gru_in = tf.concat([mfcc_net, mel_net, angular_net], axis=2)

    fw_cell_list = [tf.nn.rnn_cell.GRUCell(2048) for i in range(3)]
    bw_cell_list = [tf.nn.rnn_cell.GRUCell(2048) for i in range(3)]

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        fw_cell_list,
        bw_cell_list,
        gru_in,
        dtype=tf.float32
    )

    bi_gru = tf.layers.dropout(outputs, rate=0.5, training=is_training)

    shape = bi_gru.get_shape().as_list()  # [batch, width, 2*n_hidden]
    bi_gru_reshaped = tf.reshape(bi_gru, [-1, shape[-1]])  # [batch x width, 2*n_hidden]

    W = weightVar([2048 * 2, cfg.num_class])
    b = biasVar([cfg.num_class])
    fc_out = tf.nn.bias_add(tf.matmul(bi_gru_reshaped, W), b)

    rnn_out = tf.reshape(fc_out, [-1, 38, cfg.num_class])  # [batch, width, n_classes]

    # raw_pred = tf.argmax(tf.nn.softmax(rnn_out), axis=2)

    predictions = {
        'classed': tf.argmax(tf.nn.softmax(rnn_out), axis=2),
        'prob': tf.nn.softmax(rnn_out, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc_out)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    # mnist = tf.contrib.learn.datasets.DATASETS['mnist']('./tmp/mnist')
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./crnn_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    test_solution = data_utility.AudioPrepare()
    train_input_fn = test_solution.tf_input_fn_maker(is_training=False)

    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=2000,
    #     shuffle=False)
    # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)
