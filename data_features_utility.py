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
        '''

        :param audio_path:
        :return:
        audio_data,sampleRate
        '''
        audio_data, sampleRate = sf.read(audio_path)
        print('audio :{0}'.format(audio_path))
        print('sample rate :{0}'.format(sampleRate))
        print('shape: {0}'.format(audio_data.shape))
        return audio_data, sampleRate



    def _getMaxTDOA(self, microphoneSeparationInMetres):
        sound_speed = 340.29  # m/s
        return microphoneSeparationInMetres / sound_speed

    def _getTDOAsINSecond(self, microphoneSeparationInMetres, numTDOAs):
        max_TDOA = self._getMaxTDOA(microphoneSeparationInMetres)
        tdoa_In_Seconds = np.linspace(-max_TDOA, max_TDOA, numTDOAs)
        return tdoa_In_Seconds

    def get_angular_spectrogram(self, stereoSamples, windowSize, hopSize, fftSize=None, SampleRate=16000, num_TDOAs=128,
                                microphoneSeparationInMetres=0.05):
        '''

        :param stereoSamples:shape (2,x)
        :param windowSize:
        :param hopSize:
        :param fftSize:
        :param SampleRate:
        :param num_TDOAs: resoution
        :param microphoneSeparationInMetres:
        :return:
        '''
        if fftSize is None:
            fftSize = windowSize

        complexMixtureSpectrograms = []
        for channelIndex in range(2):
            complexMixtureSpectrograms.append(
                librosa.stft(np.squeeze(stereoSamples[channelIndex]).copy(), n_fft=fftSize, hop_length=hopSize, win_length=windowSize,center=False))
                # stft(np.squeeze(stereoSamples[channelIndex]).copy(),n_fft=fftSize,hop_length=hopSize,win_length=windowSize,center=False))

        complexMixtureSpectrograms = np.array(complexMixtureSpectrograms)

        num_channel, num_frequenies, num_time = complexMixtureSpectrograms.shape
        frequenciesInHz = np.linspace(0, SampleRate / 2, num_frequenies)

        spectralCoherenceV = complexMixtureSpectrograms[0] * complexMixtureSpectrograms[1].conj() / np.abs(
            complexMixtureSpectrograms[0]) / np.abs(complexMixtureSpectrograms[1])

        tdoasInSeconds = self._getTDOAsINSecond(microphoneSeparationInMetres, num_TDOAs)
        expJOmega = np.exp(np.outer(frequenciesInHz, -(2j * np.pi) * tdoasInSeconds))

        FREQ, TIME, TDOA = range(3)
        return np.sum(np.einsum(spectralCoherenceV, [FREQ, TIME], expJOmega, [FREQ, TDOA], [TDOA, FREQ, TIME]).real,
                      axis=1), tdoasInSeconds

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

    def feature_extract(self, audio_path, mfcc_bands=40, mel_spec_n_fft=640):
        audio_data, sample_rate = self.read_wav(audio_path)

        # chan_0 = audio_data[:, 0]
        # chan_1 = audio_data[:, 1]
        # chan_2 = audio_data[:, 2]
        # chan_3 = audio_data[:, 3]

        mfccs = []
        mels = []
        for i in range(4):
            # tt=(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)
            mfccs.append(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands))
            # mfccs.append(np.mean(librosa.feature.mfcc(y=audio_data[:, i], sr=sample_rate, n_mfcc=mfcc_bands)))
            mels.append(librosa.amplitude_to_db(
                librosa.feature.melspectrogram(y=audio_data[:, i], sr=sample_rate, n_fft=mel_spec_n_fft,
                                               hop_length=mel_spec_n_fft // 2), ref=np.max))

        return (mfccs, mels)

    def save_feature(self, dataset_dir="DCASE2018-task5-dev", feature_dir_name='features'):
        if not os.path.exists(feature_dir_name):
            os.mkdir(feature_dir_name)

        data_meta_dirs = os.scandir(os.path.join(dataset_dir, 'evaluation_setup'))

        for meta in data_meta_dirs:
            try:
                with open(meta, 'r') as f_meta:
                    # print(meta.name)
                    for line in f_meta.readlines():
                        pass
                        print(line.split()[0])
                        path, label, sess_label = line.split()

                        mfccs, mels = self.feature_extract(os.path.join(*[dataset_dir, *path.split('/')]))
                        feature_dict = {
                            'label': label,
                            'session': sess_label,
                            'mfcc': mfccs,
                            'mel': mels
                        }

                        with open(os.path.join(*[feature_dir_name,
                                                 meta.name.split('.txt')[0] + '-' + path.split('/')[1].split('.wav')[
                                                     0] + '-' + label]), 'wb') as f:
                            pickle.dump(feature_dict, f)
                        print('dumping ' + os.path.join(*[feature_dir_name, meta.name.split('.txt')[0] + '-' +
                                                          path.split('/')[1].split('.wav')[0] + '-' + label]))

            except Exception as e:
                print(e)


if __name__ == "__main__":
    test_solution = Audio_perpare()
    # test_solution.save_feature()

    # phat test
    # refsig = np.linspace(1, 10, 10)
    #
    # for i in range(0, 10):
    #     sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
    #     offset, _ = test_solution.gcc_phat(sig, refsig)
    #     print(offset)

    # angular spectrum test
    tdoa_data = "tdoa_test.wav"
    data, sr = test_solution.read_wav(tdoa_data)
    # data = data.reshape(2, -1)
    data=data.T
    angular_spec, tdoasInSeconds = test_solution.get_angular_spectrogram(data, windowSize=1024, hopSize=128,
                                                                         fftSize=1024,
                                                                         microphoneSeparationInMetres=0.086)
    angular_spec[angular_spec < 0] = 0
    # plt.imshow(angular_spec,extent=[0,10,-0.002,0.002],cmap=cm.binary)
    librosa.display.specshow(angular_spec)
    plt.show()
    meanAngularSpectrum = np.mean(angular_spec, axis=-1)
    plt.plot(tdoasInSeconds, meanAngularSpectrum)
    plt.show()
    pass
