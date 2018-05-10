import tensorflow as tf
import numpy as np
import scipy as sci
import config as cfg
import data_features_utility as data_utility


def conv_layers(input, is_training, pooling_config=None, name=None, use_bn=True, use_dropout=True):
    if pooling_config == None:
        pooling_config = [2, 2, 2]

    with tf.variable_scope('conv1_' + name):

        net = tf.layers.conv2d(
            input,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, axis=1, training=is_training)

        net = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[0], 2), padding='same')
        if use_dropout == True:
            pool1 = tf.layers.dropout(net, rate=0.5, training=is_training)

    with tf.variable_scope('conv2_' + name):
        net = tf.layers.conv2d(
            pool1,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)

        pool2 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[1], 2), padding='same')
        if use_dropout == True:
            pool2 = tf.layers.dropout(pool2, rate=0.5, training=is_training)

    with tf.variable_scope('conv3_' + name):
        net = tf.layers.conv2d(
            pool2,
            filters=32,
            kernel_size=3,
            activation=tf.nn.relu)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)

        pool3 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[2], 2), padding='same')
        if use_dropout == True:
            pool3 = tf.layers.dropout(pool3, rate=0.5, training=is_training)

    with tf.variable_scope('Reshape_cnn_' + name):
        output_shape = pool3.get_shape().as_list()  # [batch,height,width,features]
        output = tf.transpose(pool3, [0, 2, 1, 3], name='transposed')
        output = tf.reshape(output, shape=[-1, output_shape[2], output_shape[1] * output_shape[3]],
                            name='reshaped')  # [batch,width,heigth*features]
    return output


def conv_layers2(input, is_training, pooling_config=None, name=None, use_bn=True, use_dropout=True):
    if pooling_config == None:
        pooling_config = [2, 2, 2]

    with tf.variable_scope('conv1_' + name):

        net = tf.layers.conv2d(
            input,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=None)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[0], 2), padding='valid')
        if use_dropout == True:
            pool1 = tf.layers.dropout(net, rate=0.5, training=is_training)

    with tf.variable_scope('conv2_' + name):
        net = tf.layers.conv2d(
            pool1,
            filters=32,
            kernel_size=3,
            padding='same',
            activation=None)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)
        pool2 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[1], 2), padding='valid')
        if use_dropout == True:
            pool2 = tf.layers.dropout(pool2, rate=0.5, training=is_training)

    with tf.variable_scope('conv3_' + name):
        net = tf.layers.conv2d(
            pool2,
            filters=32,
            kernel_size=3,
            activation=None)
        if use_bn == True:
            net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.nn.relu(net)
        pool3 = tf.layers.max_pooling2d(net, pool_size=2, strides=(pooling_config[2], 2), padding='valid')
        if use_dropout == True:
            pool3 = tf.layers.dropout(pool3, rate=0.5, training=is_training)

    with tf.variable_scope('Reshape_cnn_' + name):
        output_shape = pool3.get_shape().as_list()  # [batch,height,width,features]
        output = tf.transpose(pool3, [0, 2, 1, 3], name='transposed')
        output = tf.reshape(output, shape=[-1, output_shape[2], output_shape[1] * output_shape[3]],
                            name='reshaped')  # [batch,width,heigth*features]
    return output


def conv_layers3(input, is_training, name=None):
    with tf.name_scope('conv1'):
        net = tf.layers.conv2d(
            inputs=input,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.name_scope('conv2'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.name_scope('conv3'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.name_scope('conv4'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.name_scope('conv5'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.name_scope('conv6'):
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=None)
        net = tf.layers.batch_normalization(net, axis=1, training=is_training)
        net = tf.nn.relu(features=net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
        net = tf.layers.dropout(net, 0.5, training=is_training)

    with tf.variable_scope('Reshape_cnn_' + name):
        output_shape = net.get_shape().as_list()  # [batch,height,width,features]
        output = tf.transpose(net, [0, 2, 1, 3], name='transposed')
        output = tf.reshape(output, shape=[-1, output_shape[2], output_shape[1] * output_shape[3]],
                            name='reshaped')  # [batch,width,heigth*features]
    return output


def model_fn(features, labels, mode):
    mels = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])
    mfccs = tf.reshape(features['mfcc'], shape=[-1, cfg.mfcc_shape[0], cfg.mfcc_shape[1], 4])
    angulars = tf.reshape(features['angular'], shape=[-1, cfg.anguler_shape[0], cfg.anguler_shape[1], 6])

    # img_scale = tf.constant([255], tf.float32)
    # tf.summary.image('mel_image', tf.cast(tf.multiply(mels, img_scale), tf.uint8))
    # tf.summary.image('mfcc_image', tf.cast(tf.multiply(mfccs[-1, :, :, 0], img_scale), tf.uint8))
    # tf.summary.image('angulars_image', tf.cast(tf.multiply(angulars[-1, :, :, 0], img_scale), tf.uint8))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # mfcc_net = conv_layers2(mfccs, is_training, pooling_config=[4, 2, 2], name='mfcc', use_bn=True, use_dropout=True)
    mel_net = conv_layers2(mels, is_training, pooling_config=[4, 4, 2], name='mel', use_bn=True, use_dropout=True)
    # angular_net = conv_layers2(angulars, is_training, pooling_config=[4, 4, 2], name='angular', use_bn=True,
    #                           use_dropout=True)
    # mel_net = conv_layers3(mels, is_training, name='mel')

    with tf.variable_scope('BiGRU'):
        # gru_in = tf.concat([mfcc_net, mel_net, angular_net], axis=2)
        # gru_in=tf.concat([mel_net, angular_net], axis=2)
        gru_in = mel_net
        # gru_in = tf.Print( mel_net,[mel_net],'debugging: ')

        # fw_cell_list = [tf.nn.rnn_cell.GRUCell(256) for _ in range(3)]
        # bw_cell_list = [tf.nn.rnn_cell.GRUCell(256) for _ in range(3)]

        fw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                                          input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(3)]
        bw_cell_list = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer),
                                          input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(3)]

        fw_cell_list = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256), state_keep_prob=0.5) for _ in
                        range(1)]
        bw_cell_list = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256), state_keep_prob=0.5) for _ in
                        range(1)]

        # fw_cell_list = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer)for _ in range(3)]
        # bw_cell_list = [tf.nn.rnn_cell.GRUCell(256, kernel_initializer=tf.orthogonal_initializer) for _ in range(3)]

        # fw_cell_list = [
        #     tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256),
        #                                   input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(3)]
        # bw_cell_list = [
        #     tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(256),
        #                                   input_keep_prob=0.5, output_keep_prob=0.5) for _ in range(3)]

        multi_rnn_fW_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
        multi_rnn_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)

        rnn_outputs, (last_state_fw, last_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_rnn_fW_cell,
            cell_bw=multi_rnn_bw_cell,
            inputs=gru_in,
            dtype=tf.float32)

        rnn_outputs_merged = tf.concat(rnn_outputs, 2)
        rnn_finial = tf.unstack(rnn_outputs_merged, rnn_outputs_merged.get_shape().as_list()[1], 1)[-1]

        # fc_out = tf.layers.dense(inputs=last_state_fw[-1] + last_state_bw[-1], units=512, activation=tf.nn.relu)
        # fc_out = tf.layers.dense(inputs=rnn_finial, units=512, activation=tf.nn.relu)
        # fc_out = tf.layers.dropout(fc_out, 0.5, training=is_training)

    # logits = tf.layers.dense(inputs=rnn_finial, units=cfg.num_class, activation=None)
    # rnn_finial=last_state_fw+last_state_bw
    logits = tf.layers.dense(inputs=rnn_finial, units=cfg.num_class, activation=None)
    logits = tf.layers.dropout(logits, 0.5, training=is_training)
    # logits = tf.nn.sigmoid(logits)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1))
    accuracy = tf.Print(accuracy, [accuracy], 'Acuracy__')
    tf.summary.scalar('train_accuracy', accuracy[1])

    predictions = {
        'classes': tf.argmax(tf.nn.softmax(logits), axis=1, name='predict_class'),
        'prob': tf.nn.softmax(logits, name='softmax_tensor'),
        # 'label':labels
        'training_accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1),
                                                 name='xxx'),
        # 'mfcc_out':mfcc_net,
        # 'mel_out':mel_net,
        # 'angular':angular_net,
        # 'output_fw':outputs_fw,
        # 'output_bw':outputs_bw,
        # 'last_state_fw':last_state_fw,
        # 'last_state_bw':last_state_bw

    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    #
    test_solution = data_utility.AudioPrepare()
    train_input_fn = test_solution.tf_input_fn_maker(is_training=True, n_epoch=10)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./crnn_model")

    tensors_to_log = {'class': 'predict_class', 'prob': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    test_solution = data_utility.AudioPrepare()
    test_input_fn = test_solution.tf_input_fn_maker(is_training=False, n_epoch=1)

    tensors_to_log = {'acc': 'accuracy', }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    eval_results = classifier.evaluate(input_fn=test_input_fn, steps=100, hooks=[logging_hook])
    print(eval_results)
