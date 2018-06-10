import matlab.engine as m_eng
import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display as rdis

import numpy as np
import scipy as sci
from multiprocessing import Process, Pool
import config as cfg

import pickle
import gzip
from tqdm import tqdm
from itertools import combinations


def worker(audio_dir: list, save_dir, process_i=0):
    def read_wav(audio_path):
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

    def _getMaxTDOA(microphoneSeparationInMetres):
        sound_speed = 340.29  # m/s
        return microphoneSeparationInMetres / sound_speed

    def _getTDOAsINSecond(microphoneSeparationInMetres, numTDOAs):
        max_TDOA = _getMaxTDOA(microphoneSeparationInMetres)
        tdoa_In_Seconds = np.linspace(-max_TDOA, max_TDOA, numTDOAs)
        return tdoa_In_Seconds

    def get_angular_spectrogram(input_signal, ref_signal, windowSize, hopSize, fftSize=None, SampleRate=16000,
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

        tdoasInSeconds = _getTDOAsINSecond(microphoneSeparationInMetres, num_TDOAs)
        expJOmega = np.exp(np.outer(frequenciesInHz, -(2j * np.pi) * tdoasInSeconds))

        FREQ, TIME, TDOA = range(3)
        angular_spectrogram = np.sum(
            np.einsum(spectralCoherenceV, [FREQ, TIME], expJOmega, [FREQ, TDOA], [TDOA, FREQ, TIME]).real,
            axis=1)

        angular_spectrogram[angular_spectrogram < 0] = 0
        meanAngularSpectrum = np.mean(angular_spectrogram, axis=-1)
        return angular_spectrogram, meanAngularSpectrum

    def feature_extract(audio_path, mfcc_bands=40, mfcc_n_fft=1024, mel_spec_n_fft=512, angular_windowsize=1024,
                        angular_n_fft=1024,
                        num_TDOA=80):  # mel_spec_n_fft=640
        audio_data, sample_rate = read_wav(audio_path)
        combs = list(combinations([i for i in range(4)], 2))

        mfccs = []
        mels = []
        angular = []
        for i in range(audio_data.shape[1]):
            # tt=(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)
            mfccs.append(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands, n_fft=mfcc_n_fft,
                                              hop_length=mfcc_n_fft // 2))
            # mfccs.append(np.mean(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)))
            mels.append(librosa.amplitude_to_db(
                librosa.feature.melspectrogram(y=audio_data[:, i], sr=sample_rate, n_fft=mel_spec_n_fft,
                                               hop_length=mel_spec_n_fft // 2), ref=np.max))

        for i, j in combs:
            angular.append(
                get_angular_spectrogram(audio_data[:, i], audio_data[:, j],
                                        SampleRate=sample_rate,
                                        windowSize=angular_windowsize,
                                        fftSize=angular_n_fft,
                                        hopSize=angular_n_fft // 2,
                                        num_TDOAs=num_TDOA)[0])
        mfccs = np.asarray(mfccs)
        mels = np.asarray(mels)
        angular = np.asarray(angular)

        paddings = np.zeros([angular.shape[0], angular.shape[1], 2])
        angular = np.concatenate([angular, paddings], 2)
        return (mfccs, mels, angular)

    for var in tqdm(audio_dir, desc='process {0}'.format(process_i)):
        mfcc, mel, angular = feature_extract(var,
                                             mfcc_bands=cfg.mfcc_bands,
                                             mfcc_n_fft=cfg.mfcc_n_fft,
                                             mel_spec_n_fft=cfg.mel_spec_n_fft,
                                             angular_windowsize=cfg.angular_windowsize,
                                             angular_n_fft=cfg.angular_n_fft,
                                             num_TDOA=cfg.num_TDOA)

        feature_dict = {
            'mfcc': mfcc,
            'mel': mel,
            'angular': angular
        }
        save_name = var.split('\\')[-1].split('.wav')[0] + '.gzip'
        with gzip.open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(feature_dict, f)


if __name__ == '__main__':
    # eng = m_eng.start_matlab()

    dataset_dir = "DCASE2018-task5-dev"
    audio_dirs = list(os.scandir(os.path.join(dataset_dir, 'audio')))
    feature_dirs = "mel_angular_mfcc"

    if not os.path.exists(feature_dirs):
        os.mkdir(feature_dirs)

    audio_path = [[] for _ in range(6)]
    for i in range(len(audio_dirs)):
        if audio_dirs[i].name.split('.')[1] == 'wav':
            audio_path[i % 6].append(audio_dirs[i].path)

    process_pool = Pool()
    for i in range(6):
        process_pool.apply_async(worker, args=(audio_path[i], feature_dirs,i))

    process_pool.close()
    process_pool.join()
    # worker(audio_path[0], feature_dirs, 0)
    print("main process ends")

    # feature_acr_stft = list(os.scandir(feature_ACR_dirs))
    # with gzip.open(feature_acr_stft[0], 'rb') as f:
    #     xx = pickle.load(f)
    #     pass

    pass
