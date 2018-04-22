import numpy as np
import librosa
import librosa.display as rosa_display
import matplotlib.pyplot as plt
import os
import glob
import sys
import pickle
import soundfile as sf
import config as cfg
from tqdm import tqdm

from itertools import combinations
import tensorflow as tf


class AudioPrepare():
    def __init__(self):
        pass

    def read_wav(self, audio_path):
        '''

        :param audio_path:
        :return:
        audio_data,sampleRate
        '''
        audio_data, sampleRate = sf.read(audio_path)
        # print('audio :{0}'.format(audio_path))
        # print('sample rate :{0}'.format(sampleRate))
        # print('shape: {0}'.format(audio_data.shape))
        return audio_data, sampleRate

    def _getMaxTDOA(self, microphoneSeparationInMetres):
        sound_speed = 340.29  # m/s
        return microphoneSeparationInMetres / sound_speed

    def _getTDOAsINSecond(self, microphoneSeparationInMetres, numTDOAs):
        max_TDOA = self._getMaxTDOA(microphoneSeparationInMetres)
        tdoa_In_Seconds = np.linspace(-max_TDOA, max_TDOA, numTDOAs)
        return tdoa_In_Seconds

    def get_angular_spectrogram_test(self, stereoSamples, windowSize, hopSize, fftSize=None, SampleRate=16000,
                                     num_TDOAs=128,
                                     microphoneSeparationInMetres=0.05):
        '''

        :param stereoSamples:shape (2,x)
        :param windowSize:
        :param hopSize:
        :param fftSize:
        :param SampleRate:
        :param num_TDOAs: resolution, nums of data to accmulate
        :param microphoneSeparationInMetres:
        :return:
        '''
        if fftSize is None:
            fftSize = windowSize

        complexMixtureSpectrograms = []
        for channelIndex in range(2):
            complexMixtureSpectrograms.append(
                librosa.stft(np.squeeze(stereoSamples[channelIndex]).copy(), n_fft=fftSize, hop_length=hopSize,
                             win_length=windowSize, center=False))

        complexMixtureSpectrograms = np.array(complexMixtureSpectrograms)

        num_channel, num_frequenies, num_time = complexMixtureSpectrograms.shape
        frequenciesInHz = np.linspace(0, SampleRate / 2, num_frequenies)

        spectralCoherenceV = complexMixtureSpectrograms[0] * complexMixtureSpectrograms[1].conj() / np.abs(
            complexMixtureSpectrograms[0]) / np.abs(complexMixtureSpectrograms[1])

        tdoasInSeconds = self._getTDOAsINSecond(microphoneSeparationInMetres, num_TDOAs)
        expJOmega = np.exp(np.outer(frequenciesInHz, -(2j * np.pi) * tdoasInSeconds))

        FREQ, TIME, TDOA = range(3)
        angular_spectrogram = np.sum(
            np.einsum(spectralCoherenceV, [FREQ, TIME], expJOmega, [FREQ, TDOA], [TDOA, FREQ, TIME]).real,
            axis=1)

        angular_spectrogram[angular_spectrogram < 0] = 0
        meanAngularSpectrum = np.mean(angular_spectrogram, axis=-1)
        return angular_spectrogram, meanAngularSpectrum

    def get_angular_spectrogram(self, input_signal, ref_signal, windowSize, hopSize, fftSize=None, SampleRate=16000,
                                num_TDOAs=128,
                                microphoneSeparationInMetres=0.05):

        if fftSize is None:
            fftSize = windowSize

        complexMixtureSpectrograms = []

        complexMixtureSpectrograms.append(
            librosa.stft(np.squeeze(input_signal).copy(), n_fft=fftSize, hop_length=hopSize,
                         win_length=windowSize, center=False))
        complexMixtureSpectrograms.append(
            librosa.stft(np.squeeze(ref_signal).copy(), n_fft=fftSize, hop_length=hopSize,
                         win_length=windowSize, center=False))

        complexMixtureSpectrograms = np.array(complexMixtureSpectrograms)

        num_channel, num_frequenies, num_time = complexMixtureSpectrograms.shape
        frequenciesInHz = np.linspace(0, SampleRate / 2, num_frequenies)

        spectralCoherenceV = complexMixtureSpectrograms[0] * complexMixtureSpectrograms[1].conj() / np.abs(
            complexMixtureSpectrograms[0]) / np.abs(complexMixtureSpectrograms[1])

        tdoasInSeconds = self._getTDOAsINSecond(microphoneSeparationInMetres, num_TDOAs)
        expJOmega = np.exp(np.outer(frequenciesInHz, -(2j * np.pi) * tdoasInSeconds))

        FREQ, TIME, TDOA = range(3)
        angular_spectrogram = np.sum(
            np.einsum(spectralCoherenceV, [FREQ, TIME], expJOmega, [FREQ, TDOA], [TDOA, FREQ, TIME]).real,
            axis=1)

        angular_spectrogram[angular_spectrogram < 0] = 0
        meanAngularSpectrum = np.mean(angular_spectrogram, axis=-1)
        return angular_spectrogram, meanAngularSpectrum

    def gcc_phat(self, sig, refsig, fs=1, max_tau=None, interp=16):
        '''
        This function computes the offset between the signal sig and the reference signal refsig
        using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
        '''

        # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
        n = sig.shape[0] + refsig.shape[0]

        # Generalized Cross Correlation Phase Transform
        SIG = np.fft.rfft(sig, n=n)
        REFSIG = np.fft.rfft(refsig, n=n)
        R = SIG * np.conj(REFSIG)

        cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

        max_shift = int(interp * n / 2)
        if max_tau:
            max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

        # find max cross correlation index
        shift = np.argmax(np.abs(cc)) - max_shift

        tau = shift / float(interp * fs)

        return tau, cc

    def feature_extract(self, audio_path, mfcc_bands=40, mel_spec_n_fft=640, angular_windowsize=1024, angular_hop=1024,
                        num_TDOA=80):
        audio_data, sample_rate = self.read_wav(audio_path)
        combs = list(combinations([i for i in range(4)], 2))
        # chan_0 = audio_data[:, 0]
        # chan_1 = audio_data[:, 1]
        # chan_2 = audio_data[:, 2]
        # chan_3 = audio_data[:, 3]

        mfccs = []
        mels = []
        angular = []
        for i in range(4):
            # tt=(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)
            mfccs.append(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands))
            # mfccs.append(np.mean(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)))
            mels.append(librosa.amplitude_to_db(
                librosa.feature.melspectrogram(y=audio_data[:, i], sr=sample_rate, n_fft=mel_spec_n_fft,
                                               hop_length=mel_spec_n_fft // 2), ref=np.max))

        for i, j in combs:
            angular.append(
                self.get_angular_spectrogram(audio_data[:, i], audio_data[:, j],
                                             windowSize=angular_windowsize,
                                             hopSize=angular_hop,
                                             num_TDOAs=num_TDOA)[0])
        mfccs = np.asarray(mfccs)
        mels = np.asarray(mels)
        angular = np.asarray(angular)
        return (mfccs, mels, angular)

    def save_feature(self, dataset_dir="DCASE2018-task5-dev", feature_dir_name='features'):
        if not os.path.exists(feature_dir_name):
            os.mkdir(feature_dir_name)

        data_meta_dirs = os.scandir(os.path.join(dataset_dir, 'evaluation_setup'))

        for meta in data_meta_dirs:
            try:
                with open(meta, 'r') as f_meta:
                    # print(meta.name)
                    for line in tqdm(f_meta.readlines()):
                        # print(line.split()[0])
                        path, label, sess_label = line.split()

                        mfccs, mels, angular = self.feature_extract(os.path.join(*[dataset_dir, *path.split('/')]))
                        feature_dict = {
                            'label': label,
                            'session': sess_label,
                            'mfcc': mfccs,
                            'mel': mels,
                            'angular': angular
                        }

                        with open(os.path.join(*[feature_dir_name,
                                                 meta.name.split('.txt')[0] + '-' + path.split('/')[1].split('.wav')[
                                                     0] + '-' + label]), 'wb') as f:
                            pickle.dump(feature_dict, f)
                        # print('dumping ' + os.path.join(*[feature_dir_name, meta.name.split('.txt')[0] + '-' +
                        #                                   path.split('/')[1].split('.wav')[0] + '-' + label]))

            except Exception as e:
                print(e)

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def save_feature_TFrecord(self, dataset_dir="DCASE2018-task5-dev", feature_dir_name='features_tfrecord'):
        if not os.path.exists(feature_dir_name):
            os.mkdir(feature_dir_name)

        data_meta_dirs = os.scandir(os.path.join(dataset_dir, 'evaluation_setup'))

        for meta in data_meta_dirs:
            # try:
                with open(meta, 'r') as f_meta:
                    # print(meta.name)
                    with tf.python_io.TFRecordWriter(
                            os.path.join(feature_dir_name, meta.name.split('.txt')[0]) + '.tfrecord') as writer:
                        # print('processing {0}'.format(meta.name.split('.txt')[0]))
                        for line in tqdm(f_meta.readlines()):
                            # print("\n" + line.split()[0])
                            path, label, sess_label = line.split()
                            # full_path = os.path.join(*[dataset_dir, *path.split('/')])

                            mfccs, mels, angular = self.feature_extract(os.path.join(*[dataset_dir, *path.split('/')]),
                                                                        mfcc_bands=cfg.mfcc_bands,
                                                                        mel_spec_n_fft=cfg.mel_spec_n_fft,
                                                                        angular_windowsize=cfg.angular_windowsize,
                                                                        angular_hop=cfg.angular_hop)
                            label_feature={
                                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[cfg.class_name2index[label]])),
                                'session': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(sess_label, encoding='utf-8')]))
                            }
                            # features = {
                            #     # 'label': self.int64_feature(cfg.class_name2index[label]),
                            #     # 'session': self.bytes_feature(bytes(sess_label, encoding='utf-8')),
                            #     'mfcc': tf.train.Feature(float_list=tf.train.FloatList(value=mfccs.flatten().tolist())),
                            #     'mel': tf.train.Feature(float_list=tf.train.FloatList(value=mels.flatten().tolist())),
                            #     'angular': tf.train.Feature(
                            #         float_list=tf.train.FloatList(value=angular.flatten().tolist()))
                            # }
                            # example = tf.train.Example(features=tf.train.Features(feature=features))
                            mfccs=[tf.train.Feature(float_list=tf.train.FloatList(value=(mfccs.flatten().tolist())))]
                            mels=[tf.train.Feature(float_list=tf.train.FloatList(value=(mels.flatten().tolist())))]
                            angular=[tf.train.Feature(float_list=tf.train.FloatList(value=(angular.flatten().tolist())))]
                            features={
                                'mfcc': tf.train.FeatureList(feature=mfccs),
                                'mel':tf.train.FeatureList(feature=mels),
                                'angular':tf.train.FeatureList(feature=angular),

                            }
                            example =tf.train.SequenceExample(
                                context=tf.train.Features(feature=label_feature),
                                feature_lists=tf.train.FeatureLists(feature_list=features)
                            )
                            serialized = example.SerializeToString()
                            writer.write(serialized)
            # except Exception as e:
            #     print(e)

    def read_feature_test(self, feature_dir_name='features'):
        pass

    def read_feature_TFrecord_test(self, feature_dir_name='features_tfrecord'):
        if not os.path.exists(feature_dir_name):
            raise Exception('feature_dir_name not exist')
        TF_Record_file = [dir.path for dir in os.scandir(feature_dir_name)]

        filename_queue=tf.train.string_input_producer(TF_Record_file,num_epochs=None,shuffle=False)

        reader=tf.TFRecordReader()
        _,serialized_example=reader.read(filename_queue)


        pass


if __name__ == "__main__":
    test_solution = AudioPrepare()
    # test_solution.save_feature()
    # test_solution.read_feature_TFrecord_test()
    test_solution.save_feature_TFrecord()


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
