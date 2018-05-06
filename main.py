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
            filters=32,
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
            filters=32,
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
            filters=32,
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
    mels = tf.reshape(features['mel'], shape=[-1, cfg.mel_shape[0], cfg.mel_shape[1], 4])
    mfccs = tf.reshape(features['mfcc'], shape=[-1, cfg.mfcc_shape[0], cfg.mfcc_shape[1], 4])
    angulars = tf.reshape(features['angular'], shape=[-1, cfg.anguler_shape[0], cfg.anguler_shape[1], 6])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    mfcc_net = conv_layers(mfccs, is_training, pooling_config=[2, 2, 2], name='mfcc')
    mel_net = conv_layers(mels, is_training, pooling_config=[4, 2, 2], name='mel')
    angular_net = conv_layers(angulars, is_training, pooling_config=[4, 2, 2], name='angular')

    with tf.variable_scope('BiGRU'):
        gru_in = tf.concat([mfcc_net, mel_net, angular_net], axis=2)

        fw_cell_list = [tf.nn.rnn_cell.GRUCell(512) for i in range(2)]
        bw_cell_list = [tf.nn.rnn_cell.GRUCell(512) for i in range(2)]

        multi_rnn_fW_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
        multi_rnn_bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)

        ((outputs_fw, outputs_bw), (last_state_fw, last_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multi_rnn_fW_cell,
            cell_bw=multi_rnn_bw_cell,
            inputs=gru_in,
            dtype=tf.float32)


        fc_out=tf.layers.dense(inputs=last_state_fw[1]+last_state_bw[1],units=1024,activation=tf.nn.relu)
        fc_out=tf.layers.dropout(fc_out,0.5,training=is_training)

        logits=tf.layers.dense(inputs=fc_out,units=cfg.num_class)

        accuracy=tf.metrics.accuracy(labels=labels, predictions=tf.argmax(tf.nn.softmax(logits), axis=1))
        tf.summary.scalar('train_accuracy',accuracy[1])


    predictions = {
        'classes': tf.argmax(tf.nn.softmax(logits), axis=1,name='predict_class'),
        'prob': tf.nn.softmax(logits, name='softmax_tensor'),
        # 'accuracy':tf.metrics.accuracy(labels=labels,predictions=tf.argmax(tf.nn.softmax(logits), axis=1),name='accuracy'),
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

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc_out)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    eval_metric={
        'accuracy':tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])
    }
    if mode ==tf.estimator.ModeKeys.EVAL:
        return  tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric)



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    test_solution = data_utility.AudioPrepare()
    train_input_fn = test_solution.tf_input_fn_maker(is_training=True,n_epoch=100)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir="./crnn_model")

    tensors_to_log = {"predictions":'softmax_tensor','predictions':'predict_class'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    test_solution = data_utility.AudioPrepare()
    test_input_fn = test_solution.tf_input_fn_maker(is_training=False,n_epoch=1)
    eval_results = classifier.evaluate(input_fn=test_input_fn,steps=1000)
    print(eval_results)
