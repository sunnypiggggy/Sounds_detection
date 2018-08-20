import numpy as np
import librosa
import librosa.display as rosa_display
import matplotlib.pyplot as plt
import os
import glob
import sys
import pickle
import gzip
import soundfile as sf
import config as cfg
from tqdm import tqdm
from functools import reduce

# from random import shuffle
import random

from itertools import combinations
import tensorflow as tf
from multiprocessing import Process, Pool


# mfcc.shape (4, 40, 313) mel.shape (4, 128, 313) angular.shape (6, 80, 311)
class AudioPrepare():

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def split_chunks(self, listx, size):
        for i in range(0, len(listx), size):
            yield listx[i:i + size]

    def save_feature_TFrecord_mutipross(self, dataset_dir="DCASE2018-task5-dev", feature_dir_name='features_tfrecordV2',
                                        train_sets=True):
        if not os.path.exists(feature_dir_name):
            os.mkdir(feature_dir_name)

        data_meta_dirs = list(os.scandir(os.path.join(dataset_dir, 'evaluation_setup')))
        if train_sets == True:
            data_meta_dirs = filter(lambda x: x.name.split('_')[1].split('.txt')[0] == 'train', data_meta_dirs)
        else:
            data_meta_dirs = filter(lambda x: x.name.split('_')[1].split('.txt')[0] == 'test', data_meta_dirs)

        data_meta_dirs = list(data_meta_dirs)
        # self.tfrecord_worker(data_meta_dirs[0].path,data_meta_dirs[0].name.split('.txt')[0],dataset_dir, feature_dir_name)

        pross_pool = Pool()
        for i in range(len(data_meta_dirs)):
            pross_pool.apply_async(self.tfrecord_worker, args=(
                data_meta_dirs[i].path, data_meta_dirs[i].name.split('.txt')[0], dataset_dir, feature_dir_name))

        pross_pool.close()
        pross_pool.join()
        print('main pross end')

    def tfrecord_worker(self, meta, fold_name, dataset_dir, feature_dir_name):
        print('start  process {0} TFrecord'.format(meta))
        with open(meta, 'r') as f_meta:
            if not os.path.exists(os.path.join(feature_dir_name, fold_name)):
                os.mkdir(os.path.join(feature_dir_name, fold_name))

            with open(meta, 'r') as f_meta:
                f_meta.seek(0)
                meta_lines = f_meta.readlines()
                random.shuffle(meta_lines)
                meta_chunk = list(self.split_chunks(meta_lines, 1000))

                for i in range(len(meta_chunk)):
                    with tf.python_io.TFRecordWriter(
                            os.path.join(feature_dir_name, fold_name, str(i) + '.tfrecord')) as writer:
                        for line in tqdm(meta_chunk[i], desc='\n' + fold_name + ' chunck ' + str(i)):
                            path, label, sess_label = line.split()
                            audio_file_name = path.split('/')[-1].split('.')[0]

                            with gzip.open(os.path.join('mel_angular_mfcc', audio_file_name + '.gzip')) as f_handle:
                                feature_dict = pickle.load(f_handle)
                                mfcc = feature_dict['mfcc']
                                mel = feature_dict['mel']
                                angular = feature_dict['angular']

                            with gzip.open(os.path.join('wavelets', audio_file_name + '.gzip')) as f_handle:
                                feature_dict = pickle.load(f_handle)
                                morse = feature_dict['morse']
                                bump = feature_dict['bump']

                            with gzip.open(os.path.join('gfcc', audio_file_name + '.gzip')) as f_handle:
                                feature_dict = pickle.load(f_handle)
                                gfcc = feature_dict['gfcc']

                            with gzip.open(os.path.join('ACR_stft', audio_file_name + '.gzip')) as f_handle:
                                feature_dict = pickle.load(f_handle)
                                acr_stft = feature_dict['acr_stft']

                            features = {
                                'label': self.int64_feature(cfg.class_name2index[label]),
                                'session': self.bytes_feature(bytes(sess_label, encoding='utf-8')),
                                'mfcc': tf.train.Feature(float_list=tf.train.FloatList(value=mfcc.flatten().tolist())),
                                'mfcc_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(mfcc.shape))),
                                'mel': tf.train.Feature(float_list=tf.train.FloatList(value=mel.flatten().tolist())),
                                'mel_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(mel.shape))),
                                'angular': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=angular.flatten().tolist())),
                                'angular_shape': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=list(angular.shape))),

                                'morse': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=morse.flatten().tolist())),
                                'morse_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(morse.shape))),

                                'bump': tf.train.Feature(float_list=tf.train.FloatList(value=bump.flatten().tolist())),
                                'bump_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(bump.shape))),

                                'gfcc': tf.train.Feature(float_list=tf.train.FloatList(value=gfcc.flatten().tolist())),
                                'gfcc_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(gfcc.shape))),

                                'acr_stft': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=acr_stft.flatten().tolist())),
                                'acr_stft_shape': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=list(acr_stft.shape))),

                            }
                            example = tf.train.Example(features=tf.train.Features(feature=features))
                            serialized = example.SerializeToString()
                            writer.write(serialized)

    def save_eval_data(self, feature_dir_name='evalsets'):
        if not os.path.exists(feature_dir_name):
            os.mkdir(feature_dir_name)

        mel_angular_mfcc_dirs = list(os.scandir('mel_angular_mfcc'))
        wavelets_dirs = list(os.scandir('wavelets'))
        gfcc_dirs = list(os.scandir('gfcc'))
        ACR_stft_dirs = list(os.scandir('ACR_stft'))

        assert len(mel_angular_mfcc_dirs) == len(wavelets_dirs) == len(gfcc_dirs) == len(ACR_stft_dirs)

        with tf.python_io.TFRecordWriter(os.path.join(feature_dir_name, 'eval' + '.tfrecord'))as writer:
            for i in tqdm(range(len(mel_angular_mfcc_dirs))):
                with  gzip.open(os.path.join('mel_angular_mfcc', str(i + 1) + '.gzip')) as f_handle:
                    feature_dict = pickle.load(f_handle)
                    mfcc = feature_dict['mfcc']
                    mel = feature_dict['mel']
                    angular = feature_dict['angular']

                with gzip.open(os.path.join('wavelets', str(i + 1) + '.gzip')) as f_handle:
                    feature_dict = pickle.load(f_handle)
                    morse = feature_dict['morse']
                    bump = feature_dict['bump']

                with gzip.open(os.path.join('gfcc', str(i + 1) + '.gzip')) as f_handle:
                    feature_dict = pickle.load(f_handle)
                    gfcc = feature_dict['gfcc']

                with gzip.open(os.path.join('ACR_stft', str(i + 1) + '.gzip')) as f_handle:
                    feature_dict = pickle.load(f_handle)
                    acr_stft = feature_dict['acr_stft']

                features = {
                    'label': self.int64_feature(0),
                    'file': self.int64_feature(i + 1),

                    'mfcc': tf.train.Feature(float_list=tf.train.FloatList(value=mfcc.flatten().tolist())),
                    'mfcc_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(mfcc.shape))),
                    'mel': tf.train.Feature(float_list=tf.train.FloatList(value=mel.flatten().tolist())),
                    'mel_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(mel.shape))),
                    'angular': tf.train.Feature(
                        float_list=tf.train.FloatList(value=angular.flatten().tolist())),
                    'angular_shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=list(angular.shape))),

                    'morse': tf.train.Feature(
                        float_list=tf.train.FloatList(value=morse.flatten().tolist())),
                    'morse_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(morse.shape))),

                    'bump': tf.train.Feature(float_list=tf.train.FloatList(value=bump.flatten().tolist())),
                    'bump_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(bump.shape))),

                    'gfcc': tf.train.Feature(float_list=tf.train.FloatList(value=gfcc.flatten().tolist())),
                    'gfcc_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(gfcc.shape))),

                    'acr_stft': tf.train.Feature(
                        float_list=tf.train.FloatList(value=acr_stft.flatten().tolist())),
                    'acr_stft_shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=list(acr_stft.shape))),
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                serialized = example.SerializeToString()
                writer.write(serialized)

    def tf_record_prase_function(self, example_proto):

        features = {
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
            'session': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'mfcc': tf.VarLenFeature(dtype=tf.float32),
            'mfcc_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'mel': tf.VarLenFeature(dtype=tf.float32),
            'mel_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'angular': tf.VarLenFeature(dtype=tf.float32),
            'angular_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'morse': tf.VarLenFeature(dtype=tf.float32),
            'morse_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'bump': tf.VarLenFeature(dtype=tf.float32),
            'bump_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'gfcc': tf.VarLenFeature(dtype=tf.float32),
            'gfcc_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'acr_stft': tf.VarLenFeature(dtype=tf.float32),
            'acr_stft_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

        }

        parsed_features = tf.parse_single_example(
            example_proto,
            features=features
        )
        mfcc = tf.sparse_tensor_to_dense(parsed_features['mfcc'])
        mfcc = tf.reshape(mfcc, parsed_features['mfcc_shape'])
        mfcc = tf.transpose(mfcc, [1, 2, 0])

        mel = tf.sparse_tensor_to_dense(parsed_features['mel'])
        mel = tf.reshape(mel, parsed_features['mel_shape'])
        mel = tf.transpose(mel, [1, 2, 0])

        angular = tf.sparse_tensor_to_dense(parsed_features['angular'])
        angular = tf.reshape(angular, parsed_features['angular_shape'])
        angular = tf.transpose(angular, [1, 2, 0])

        morse = tf.sparse_tensor_to_dense(parsed_features['morse'])
        morse = tf.reshape(morse, parsed_features['morse_shape'])
        morse = tf.transpose(morse, [1, 2, 0])

        bump = tf.sparse_tensor_to_dense(parsed_features['bump'])
        bump = tf.reshape(bump, parsed_features['bump_shape'])
        bump = tf.transpose(bump, [1, 2, 0])

        gfcc = tf.sparse_tensor_to_dense(parsed_features['gfcc'])
        gfcc = tf.reshape(gfcc, parsed_features['gfcc_shape'])
        gfcc = tf.transpose(gfcc, [1, 2, 0])

        acr_stft = tf.sparse_tensor_to_dense(parsed_features['acr_stft'])
        acr_stft = tf.reshape(acr_stft, parsed_features['acr_stft_shape'])
        acr_stft = tf.transpose(acr_stft, [1, 2, 0])

        label = tf.cast(parsed_features['label'], tf.int32)

        return {'mfcc': mfcc, 'mel': mel, 'angular': angular, 'morse': morse, 'bump': bump, 'gfcc': gfcc,
                'acr_stft': acr_stft}, label

    def tf_record_eval_prase_function(self, example_proto):

        features = {
            'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
            'file': tf.FixedLenFeature(shape=(), dtype=tf.int64),

            'mfcc': tf.VarLenFeature(dtype=tf.float32),
            'mfcc_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'mel': tf.VarLenFeature(dtype=tf.float32),
            'mel_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'angular': tf.VarLenFeature(dtype=tf.float32),
            'angular_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'morse': tf.VarLenFeature(dtype=tf.float32),
            'morse_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'bump': tf.VarLenFeature(dtype=tf.float32),
            'bump_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'gfcc': tf.VarLenFeature(dtype=tf.float32),
            'gfcc_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

            'acr_stft': tf.VarLenFeature(dtype=tf.float32),
            'acr_stft_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),

        }

        parsed_features = tf.parse_single_example(
            example_proto,
            features=features
        )
        mfcc = tf.sparse_tensor_to_dense(parsed_features['mfcc'])
        mfcc = tf.reshape(mfcc, parsed_features['mfcc_shape'])
        mfcc = tf.transpose(mfcc, [1, 2, 0])

        mel = tf.sparse_tensor_to_dense(parsed_features['mel'])
        mel = tf.reshape(mel, parsed_features['mel_shape'])
        mel = tf.transpose(mel, [1, 2, 0])

        angular = tf.sparse_tensor_to_dense(parsed_features['angular'])
        angular = tf.reshape(angular, parsed_features['angular_shape'])
        angular = tf.transpose(angular, [1, 2, 0])

        morse = tf.sparse_tensor_to_dense(parsed_features['morse'])
        morse = tf.reshape(morse, parsed_features['morse_shape'])
        morse = tf.transpose(morse, [1, 2, 0])

        bump = tf.sparse_tensor_to_dense(parsed_features['bump'])
        bump = tf.reshape(bump, parsed_features['bump_shape'])
        bump = tf.transpose(bump, [1, 2, 0])

        gfcc = tf.sparse_tensor_to_dense(parsed_features['gfcc'])
        gfcc = tf.reshape(gfcc, parsed_features['gfcc_shape'])
        gfcc = tf.transpose(gfcc, [1, 2, 0])

        acr_stft = tf.sparse_tensor_to_dense(parsed_features['acr_stft'])
        acr_stft = tf.reshape(acr_stft, parsed_features['acr_stft_shape'])
        acr_stft = tf.transpose(acr_stft, [1, 2, 0])

        label = tf.cast(parsed_features['label'], tf.int32)
        file = tf.cast(parsed_features['file'], tf.int32)

        return {'mfcc': mfcc, 'mel': mel, 'angular': angular, 'morse': morse, 'bump': bump, 'gfcc': gfcc,
                'acr_stft': acr_stft, 'file': file}, label
    # def tf_record_eval_prase_function(self, example_proto):
    #
    #     features = {
    #         'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    #         'file': tf.FixedLenFeature(shape=(), dtype=tf.string),
    #
    #         'mfcc': tf.VarLenFeature(dtype=tf.float32),
    #         'mfcc_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #         'mel': tf.VarLenFeature(dtype=tf.float32),
    #         'mel_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #         'angular': tf.VarLenFeature(dtype=tf.float32),
    #         'angular_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #         'morse': tf.VarLenFeature(dtype=tf.float32),
    #         'morse_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #         'bump': tf.VarLenFeature(dtype=tf.float32),
    #         'bump_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #         'gfcc': tf.VarLenFeature(dtype=tf.float32),
    #         'gfcc_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #         'acr_stft': tf.VarLenFeature(dtype=tf.float32),
    #         'acr_stft_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
    #
    #     }
    #
    #     parsed_features = tf.parse_single_example(
    #         example_proto,
    #         features=features
    #     )
    #     mfcc = tf.sparse_tensor_to_dense(parsed_features['mfcc'])
    #     mfcc = tf.reshape(mfcc, parsed_features['mfcc_shape'])
    #     mfcc = tf.transpose(mfcc, [1, 2, 0])
    #
    #     mel = tf.sparse_tensor_to_dense(parsed_features['mel'])
    #     mel = tf.reshape(mel, parsed_features['mel_shape'])
    #     mel = tf.transpose(mel, [1, 2, 0])
    #
    #     angular = tf.sparse_tensor_to_dense(parsed_features['angular'])
    #     angular = tf.reshape(angular, parsed_features['angular_shape'])
    #     angular = tf.transpose(angular, [1, 2, 0])
    #
    #     morse = tf.sparse_tensor_to_dense(parsed_features['morse'])
    #     morse = tf.reshape(morse, parsed_features['morse_shape'])
    #     morse = tf.transpose(morse, [1, 2, 0])
    #
    #     bump = tf.sparse_tensor_to_dense(parsed_features['bump'])
    #     bump = tf.reshape(bump, parsed_features['bump_shape'])
    #     bump = tf.transpose(bump, [1, 2, 0])
    #
    #     gfcc = tf.sparse_tensor_to_dense(parsed_features['gfcc'])
    #     gfcc = tf.reshape(gfcc, parsed_features['gfcc_shape'])
    #     gfcc = tf.transpose(gfcc, [1, 2, 0])
    #
    #     acr_stft = tf.sparse_tensor_to_dense(parsed_features['acr_stft'])
    #     acr_stft = tf.reshape(acr_stft, parsed_features['acr_stft_shape'])
    #     acr_stft = tf.transpose(acr_stft, [1, 2, 0])
    #
    #
    #     return {'mfcc': mfcc, 'mel': mel, 'angular': angular, 'morse': morse, 'bump': bump, 'gfcc': gfcc,
    #             'acr_stft': acr_stft}
    def tf_input_fn_maker(self, feature_dir_name='features_tfrecordV2', is_training=True, n_epoch=1):
        data_dir = os.scandir(feature_dir_name)

        if is_training:
            data_folders = list(filter(lambda x: x.name.split('_')[1] == 'train', data_dir))
        else:
            data_folders = list(filter(lambda x: x.name.split('_')[1] == 'test', data_dir))

        print('Input datasets :{dataset} Training mode {mode} '.format(dataset=[x.name for x in data_folders],
                                                                       mode=is_training))

        data_folders = [list(os.scandir(x)) for x in data_folders]
        data_path = reduce(lambda x, y: x + y, data_folders)
        data_path = [x.path for x in data_path]
        dataset = tf.data.TFRecordDataset(data_path)

        dataset = dataset.map(self.tf_record_prase_function)
        dataset = dataset.shuffle(buffer_size=50)
        dataset = dataset.batch(32)
        dataset = dataset.repeat(n_epoch)

        def input_fn():
            iterator = dataset.make_one_shot_iterator()
            feature, label = iterator.get_next()
            return feature, label

        return input_fn

    def tf_input_fn_maker_predict(self, feature_dir_name='features_tfrecordV2', n_epoch=1):
        data_dir = os.scandir(feature_dir_name)

        data_folders = list(filter(lambda x: x.name.split('_')[1] == 'test', data_dir))

        print('Input datasets :{dataset} Predict'.format(dataset=[x.name for x in data_folders]))

        data_folders = [list(os.scandir(x)) for x in data_folders]
        data_path = reduce(lambda x, y: x + y, data_folders)
        data_path = [x.path for x in data_path]
        dataset = tf.data.TFRecordDataset(data_path)

        dataset = dataset.map(self.tf_record_prase_function)
        # dataset = dataset.shuffle(buffer_size=1001)
        dataset = dataset.batch(10)
        dataset = dataset.repeat(n_epoch)

        def input_fn():
            iterator = dataset.make_one_shot_iterator()
            feature, label = iterator.get_next()
            return feature, label

        return input_fn

    def tf_input_fn_maker_eval(self, feature_dir_name='evalsets'):
        data_dir = os.scandir(feature_dir_name)

        data_path = [x.path for x in list(data_dir)]
        dataset = tf.data.TFRecordDataset(data_path)

        dataset = dataset.map(self.tf_record_eval_prase_function)
        dataset = dataset.batch(1)
        dataset = dataset.repeat(1)

        def input_fn():
            iterator = dataset.make_one_shot_iterator()
            feature, label = iterator.get_next()
            # feature = iterator.get_next()
            return feature, label
            # return feature

        return input_fn


if __name__ == "__main__":
    test_solution = AudioPrepare()

    test_solution.save_eval_data()

    # sess = tf.InteractiveSession()
    # predict_input_fn = test_solution.tf_input_fn_maker_predict()
    # X, Y = predict_input_fn()

    # while True:
    # with open(".\data_labels.txt", 'w+') as f:
    #     # for _ in range(10):
    #     while True:
    #         try:
    #             tt = sess.run(Y)
    #
    #             for x in tt:
    #                 f.write(str(x) + '\n')
    #                 print(x)
    #         except tf.errors.OutOfRangeError:
    #             print("End of dataset")
    #             break

    # sess = tf.InteractiveSession()
    # # predict_input_fn = test_solution.tf_input_fn_maker_predict()
    # predict_input_fn = test_solution.tf_input_fn_maker_eval()
    # X= predict_input_fn()
    #
    # while True:
    #     try:
    #         # feature, label = sess.run([X])
    #         feature= sess.run([X])
    #         rosa_display.specshow(feature['mel'][0, :, :, 0])
    #         plt.show()
    #
    #     except tf.errors.OutOfRangeError:
    #         print("End of dataset")
    #         break

    # test_solution.tf_feature_dataset()
    # test_solution.save_feature()

    # test_solution.read_feature_TFrecord_test()

    # test_solution.save_feature_TFrecord()

    # test_solution.save_feature_TFrecord_mutipross(train_sets=False)
    # dataset=test_solution.tf_get_dataset(is_training=False)

    # train_iput_fn = test_solution.tf_input_fn_maker(dataset)

    # sess = tf.InteractiveSession()
    # while True:
    #     # for _ in range(10):
    #     # input_fn = test_solution.tf_input_fn_maker_test()
    #     input_fn = test_solution.tf_input_fn_maker()
    #     X, Y = input_fn()
    #     try:
    #         with open(".\data_labels.txt", 'w') as f:
    #             tt = sess.run(Y)
    #             print(tt)
    #             f.write(str(tt) + '\n')
    #         # feature ,label= sess.run([X,Y])
    #         # rosa_display.specshow(feature['mel'][0, :, :, 0])
    #         # plt.show()
    #         # label = sess.run(next_element['label'])
    #         # mfcc= sess.run(next_element['mfcc'])
    #         # print(mfcc)
    #         # print(label)
    #     except tf.errors.OutOfRangeError:
    #         print("End of dataset")
    #         break

    # feature_dir_name = 'features_tfrecord'
    # if not os.path.exists(feature_dir_name):
    #     raise Exception('feature_dir_name not exist')
    # TF_Record_file = [dir.path for dir in os.scandir(feature_dir_name)]
    # dataset = tf.data.TFRecordDataset(TF_Record_file)
    #
    # def _parse_function(example_proto):
    #     features = {
    #         'label': tf.FixedLenFeature(shape=(),dtype=tf.int64),
    #         'session': tf.FixedLenFeature(shape=(),dtype=tf.string),
    #         'mfcc': tf.VarLenFeature(dtype=tf.float32),
    #         'mfcc_shape': tf.FixedLenFeature(shape=(3,),dtype=tf.int64),
    #         'mel': tf.VarLenFeature(dtype=tf.float32),
    #         'mel_shape': tf.FixedLenFeature(shape=(3,),dtype=tf.int64),
    #         'angular': tf.VarLenFeature(dtype=tf.float32),
    #         'angular_shape': tf.FixedLenFeature(shape=(3,),dtype=tf.int64)
    #     }
    #     parsed_features = tf.parse_single_example(
    #         example_proto,
    #         features=features
    #     )
    #     parsed_features['mfcc']=tf.sparse_tensor_to_dense(parsed_features['mfcc'])
    #     parsed_features['mfcc'] = tf.reshape(parsed_features['mfcc'], parsed_features['mfcc_shape'])
    #     parsed_features['mfcc'] =tf.transpose(parsed_features['mfcc'],[1,2,0])
    #
    #     parsed_features['mel'] = tf.sparse_tensor_to_dense(parsed_features['mel'])
    #     parsed_features['mel'] = tf.reshape(parsed_features['mel'], parsed_features['mel_shape'])
    #     parsed_features['mel'] = tf.transpose(parsed_features['mel'], [1, 2, 0])
    #
    #     parsed_features['angular'] = tf.sparse_tensor_to_dense(parsed_features['angular'])
    #     parsed_features['angular'] = tf.reshape(parsed_features['angular'], parsed_features['angular_shape'])
    #     parsed_features['angular'] = tf.transpose(parsed_features['angular'], [1, 2, 0])
    #     return parsed_features
    #
    # newdataset = dataset.map(_parse_function)
    # iterator = newdataset.make_one_shot_iterator()
    # next_element = iterator.get_next()
    #
    # sess=tf.InteractiveSession()
    #
    #
    #
    # while True:
    #     try:
    #         mfcc,label=sess.run([next_element['mfcc'],next_element['label']])
    #         # label = sess.run(next_element['label'])
    #         # mfcc= sess.run(next_element['mfcc'])
    #         # print(mfcc)
    #         print(label)
    #     except tf.errors.OutOfRangeError:
    #         print("End of dataset")
    #         break

    # test_solution.save_feature_multipross()

    # save_feature_multipross()
    # phat test
    # refsig = np.linspace(1, 10, 10)
    #
    # for i in range(0, 10):
    #     sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
    #     offset, _ = test_solution.gcc_phat(sig, refsig)
    #     print(offset)

    # angular spectrum test
    # tdoa_data = "tdoa_test.wav"
    # data, sr = test_solution.read_wav(tdoa_data)
    # # data = data.reshape(2, -1)
    # data = data.T
    # angular_spec = test_solution.get_angular_spectrogram(data, windowSize=1024, hopSize=128,
    #                                                      fftSize=1024,
    #                                                      microphoneSeparationInMetres=0.086)
    # angular_spec[angular_spec < 0] = 0
    # librosa.display.specshow(angular_spec)
    # plt.show()
    # meanAngularSpectrum = np.mean(angular_spec, axis=-1)
    # plt.plot(meanAngularSpectrum)
    # plt.show()

    # test_solution.save_feature_TFrecord()

    # test_solution.save_feature()

    # Tfrecord  writer test
    # TF_writer1 = tf.python_io.TFRecordWriter('1.tfrecord')
    # TF_writer2=tf.python_io
    # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # frames = [[1], [1, 3], [1, 2, 3, 4], [1], [1, 3], [1, 2, 3, 4], [1], [1, 3], [1, 2, 3, 4]]
    #
    # for i in tqdm(range(len(labels))):
    #     label = labels[i]
    #     frame = frames[i]
    #
    #     # label_feature = tf.train.Feature(int64_list=test_solution.int64_feature(label))
    #     label_feature = test_solution.int64_feature(label)
    #     frame_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[frame_])) for frame_ in frame]
    #
    #     seq_example = tf.train.SequenceExample(
    #         context=tf.train.Features(
    #             feature={
    #                 'label': label_feature
    #             }
    #         ),
    #         feature_lists=tf.train.FeatureLists(feature_list={
    #             "frame": tf.train.FeatureList(feature=frame_feature),
    #         })
    #     )
    #     serialized = seq_example.SerializeToString()
    #     TF_writer1.write(serialized)
    #
    # # TFRecord reader Test
    # Queue_Capcity = 100
    # SHFFLE_MIN_AFTER_DEQEUE = Queue_Capcity // 5
    #
    # tfReader=tf.TFRecordReader()
    #
    # def shuffle_inputs(input_tensors, capity, min_after_dequeue, num_threads):
    #     shuffle_queue = tf.RandomShuffleQueue(
    #         capity, min_after_dequeue, dtypes=[t.dtype for t in input_tensors])
    #     enqueue_op = shuffle_queue.enqueue(input_tensors)
    #     runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * num_threads)
    #     tf.train.add_queue_runner(runner)
    #     output_tensors =shuffle_queue.dequeue()
    #
    #     for i in range(len(input_tensors)):
    #         output_tensors[i].set_shape(input_tensors[i].shape)
    #
    #     return output_tensors
    #
    # def get_padded_batch(file_list,batch_size,num_enqueuing_thread=4,shuffle=False):
    #     """Reads batches of SequenceExamples from TFRecords and pads them.
    #      Can deal with variable length SequenceExamples by padding each batch to the
    #      length of the longest sequence with zeros.
    #      Args:
    #        file_list: A list of paths to TFRecord files containing SequenceExamples.
    #        batch_size: The number of SequenceExamples to include in each batch.
    #        num_enqueuing_threads: The number of threads to use for enqueuing
    #            SequenceExamples.
    #        shuffle: Whether to shuffle the batches.
    #      Returns:
    #        labels: A tensor of shape [batch_size] of int64s.
    #        frames: A tensor of shape [batch_size, num_steps] of floats32s. note that
    #            num_steps is the max time_step of all the tensors.
    #      Raises:
    #        ValueError: If `shuffle` is True and `num_enqueuing_threads` is less than 2.
    #      """
    #     file_queue=tf.train.string_input_producer(file_list)
    #     reader=tf.TFRecordReader()
    #     _,serialized_example =reader.read(file_queue)
    #
    #     context_feature={
    #         "label":tf.FixedLenFeature([],dtype=tf.int64)
    #     }
    #     sequence_feature={
    #         "frame":tf.FixedLenFeature([],dtype=tf.int64)
    #     }

    pass
