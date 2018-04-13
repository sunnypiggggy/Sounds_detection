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


class Audio_perpare():
    def __init__(self):
        pass

    def read_wav(self, audio_path):
        audio_data, sampleRate = sf.read(audio_path)
        print('sample rate :{0}'.format(sampleRate))
        print('shape: {0}'.format(audio_data.shape))
        return audio_data, sampleRate

    def feature_extract(self, audio_path, mfcc_bands=40, mel_spec_n_fft=640):
        audio_data, sample_rate = self.read_wav(audio_path)

        # chan_0 = audio_data[:, 0]
        # chan_1 = audio_data[:, 1]
        # chan_2 = audio_data[:, 2]
        # chan_3 = audio_data[:, 3]

        mfccs = []
        mels = []
        for i in range(4):
            mfccs.append(np.mean(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)))
            mels.append(librosa.amplitude_to_db(librosa.feature.melspectrogram(y=audio_data[:, i], sr=sample_rate, n_fft=mel_spec_n_fft,
                                                       hop_length=mel_spec_n_fft // 2)),ref=np.max)

        return (mfccs, mels)

    def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
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

    def save_feature(self, dataset_dir="DCASE2018-task5-dev", feature_dir_name='features'):
        feature_dir = os.path.join(dataset_dir, feature_dir_name)
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)

        data_meta_dirs = os.scandir(os.path.join(dataset_dir, 'evaluation_setup'))

        for meta in data_meta_dirs:
            try:
                # wav_files = []
                # label = []
                # sess_label = []
                with open(meta, 'r') as f_meta:
                    for line in f_meta.readlines():
                        print(line.split()[0])
                        path,label,sess_label=line.split()

                        mfccs,mels=self.feature_extract(path)


            except Exception as e:
                print(e)


        # rootPath = os.getcwd()


if __name__ == "__main__":
    test_solution = Audio_perpare()
    # test_audio = ""
    #
    # test_solution.read_wav(test_audio)
    test_solution.save_feature()
    pass
