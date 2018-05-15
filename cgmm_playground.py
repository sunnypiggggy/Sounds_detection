import soundfile as sf
import librosa
import librosa.display as rdis
import numpy as np
import math
import os
import matplotlib.pylab as plt


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


class CGMM_MVDR():
    def _stab(self, mat, theta, num_channels):
        d = np.power(10., np.arange(-6, 0, 1))
        for i in range(6):
            if (1 / np.linalg.cond(mat)) > theta:
                break
            mat = mat + d[i] * np.eye(num_channels)
        return mat

    def __init__(self, path, fft_length=1024, hop_length=256):
        self.theta = 10e-6
        self.audio_path = path
        self.fft_length = fft_length
        self.hop_length = hop_length

        self.audio_data, self.sample_rate = self.read_wav(audio_path=self.audio_path)
        self.sample_len, self.num_channel = self.audio_data.shape

        # self.stft_data = self.compute_stft(self.audio_data, self.sample_rate)
        self.stft_data = self.compute_stft(audio_data=self.audio_data, fft_length=self.fft_length,
                                           hop_length=self.hop_length)
        self.stft_data = self.stft_data.transpose([0, 2, 1])  # shape (num_channel,num_frames,num_bins)

        self.num_channel, self.num_frames, self.num_bins = self.stft_data.shape
        # CGMM parameters
        self.lambda_noise = np.zeros([self.num_frames, self.num_bins], dtype=np.double)
        self.lambda_noisy = np.zeros([self.num_frames, self.num_bins], dtype=np.double)
        self.phi_noise = np.ones([self.num_frames, self.num_bins], dtype=np.double)
        self.phi_noisy = np.ones([self.num_frames, self.num_bins], dtype=np.double)
        self.R_noise = np.zeros([self.num_channel, self.num_channel, self.num_bins], dtype=np.complex64)
        self.R_noisy = np.zeros([self.num_channel, self.num_channel, self.num_bins], dtype=np.complex64)

        for f in range(self.num_bins):
            self.R_noisy[:, :, f] = np.dot(self.stft_data[:, :, f], self.stft_data[:, :, f].T) / self.num_frames
            self.R_noise[:, :, f] = np.eye(self.num_channel, self.num_channel)

        # precompute  y ^ H * y
        self.yyh = np.zeros([self.num_channel, self.num_channel, self.num_frames, self.num_bins], dtype=np.double)
        for f in range(self.num_bins):
            for t in range(self.num_frames):
                self.yyh[:, :, t, f] = np.outer(self.stft_data[:, t, f], self.stft_data[:, t, f])

        # init phi
        for f in range(self.num_bins):
            self.R_noisy_onbin = self._stab(self.R_noisy[:, :, f], self.theta, self.num_channel)
            self.R_noise_onbin = self._stab(self.R_noise[:, :, f], self.theta, self.num_channel)

            self.R_noisy_inv = np.linalg.inv(self.R_noisy_onbin)
            self.R_noise_inv = np.linalg.inv(self.R_noise_onbin)

            for t in range(self.num_frames):
                corre = self.yyh[:, :, t, f]
                self.phi_noise[t, f] = np.real(np.trace(np.dot(corre, self.R_noise_inv)) / self.num_channel)
                self.phi_noisy[t, f] = np.real(np.trace(np.dot(corre, self.R_noisy_inv)) / self.num_channel)

    def __str__(self):
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

    def compute_stft(self, audio_data, fft_length=1024, hop_length=256):
        stft_data = []
        # audio_len, num_channel = audio_data.shape
        for i in range(self.num_channel):
            stft_data.append(librosa.stft(audio_data[:, i], n_fft=fft_length, hop_length=hop_length))
        return np.asarray(stft_data)

    def train_CGMM(self, iter=5):
        self.p_noise = np.ones([self.num_frames, self.num_bins], dtype=np.double)
        self.p_noisy = np.ones([self.num_frames, self.num_bins], dtype=np.double)
        for i in range(iter):
            for f in range(self.num_bins):
                self.R_noisy_onbin = self._stab(self.R_noisy[:, :, f], self.theta, self.num_channel)
                self.R_noise_onbin = self._stab(self.R_noise[:, :, f], self.theta, self.num_channel)

                self.R_noisy_inv = np.linalg.inv(self.R_noisy_onbin)
                self.R_noise_inv = np.linalg.inv(self.R_noise_onbin)

                self.R_noisy_accu = np.zeros([self.num_channel, self.num_channel], dtype=np.double)
                self.R_noise_accu = np.zeros([self.num_channel, self.num_channel], dtype=np.double)

                for t in range(self.num_frames):
                    corre = self.yyh[:, :, t, f]
                    obs = self.stft_data[:, t, f]

                    # update lambda
                    # k_noise = obs.T * (self.R_noise_inv_inv / self.phi_noise[t, f]) * obs
                    k_noise = np.linalg.multi_dot([obs.T, (self.R_noise_inv/ self.phi_noise[t, f]), obs])
                    det_noise = np.linalg.det(np.dot(self.phi_noise[t, f], self.R_noise_onbin)) * np.pi
                    self.p_noise[t, f] = np.real(np.exp(-k_noise) / det_noise) + self.theta  # +theta: avoid NAN

                    # k_noisy = obs.T * (self.R_noisy_inv / self.phi_noisy[t, f]) * obs
                    k_noisy = np.linalg.multi_dot([obs.T, (self.R_noisy_inv/ self.phi_noisy[t, f]), obs])
                    det_noisy = np.linalg.det(np.dot(self.phi_noisy[t, f], self.R_noisy_onbin)) * np.pi
                    self.p_noisy[t, f] = np.real(np.exp(-k_noisy) / det_noisy) + self.theta  # +theta: avoid NAN

                    self.lambda_noise[t, f] = self.p_noise[t, f] / (self.p_noise[t, f] + self.p_noisy[t, f])
                    self.lambda_noisy[t, f] = self.p_noisy[t, f] / (self.p_noise[t, f] + self.p_noisy[t, f])

                    # update phi
                    self.phi_noise[t, f] = np.real(np.trace(np.dot(corre, self.R_noise_inv)) / self.num_channel)
                    self.phi_noisy[t, f] = np.real(np.trace(np.dot(corre, self.R_noisy_inv)) / self.num_channel)

                    # accu R
                    self.R_noise_accu = self.R_noise_accu + self.lambda_noise[t, f] / self.phi_noise[t, f] * corre
                    self.R_noisy_accu = self.R_noisy_accu + self.lambda_noisy[t, f] / self.phi_noisy[t, f] * corre
                # update R
                self.R_noise[:, :, f] = self.R_noise_accu / np.sum(self.lambda_noise[:, f])
                self.R_noisy[:, :, f] = self.R_noisy_accu / np.sum(self.lambda_noisy[:, f])

            Qn = np.sum(np.sum(self.lambda_noise * np.log(self.phi_noise))) / (self.num_frames * self.num_bins)

            Qx = np.sum(np.sum(self.lambda_noisy * np.log(self.phi_noisy))) / (self.num_frames * self.num_bins)
            print('num of it{num_it} Qn {Qn} + Qx {Qx} = sum {sum}'.format(num_it=i, Qn=Qn, Qx=Qx, sum=Qn + Qx))

        self.R_xn = self.R_noisy
        # get Rn, reference to eq.4
        self.R_n = np.zeros([self.num_channel, self.num_channel, self.num_bins])
        for f in range(self.num_bins):
            for t in range(self.num_frames):
                self.R_n[:, :, f] = self.R_n[:, :, f] + self.lambda_noise[t, f] * self.yyh[:, :, t, f]
            self.R_n[:, :, f] = self.R_n[:, :, f] / (np.sum(self.lambda_noise[:, f]))
        self.R_x = self.R_xn - self.R_n

    def apply_mvdr(self, save_path):
        self.enhanced_spec = np.zeros([self.num_frames, self.num_bins])

        for f in range(self.num_bins):
            # using Rx to estimate steer vector
            vector, value = np.linalg.eig(self.R_x[:, :, f])
            steer_vector = vector[:, 1]

            if (1 / np.linalg.cond(self.R_n[:, :, f])) < self.theta:
                self.R_n[:, :, f] = self.R_n[:, :, ] + self.theta * np.eye(self.num_channel)  # ?????????????????????

            # feed Rn into MVDR
            Rn_inv = np.linalg.inv(self.R_n[:, :, f])
            w = np.linalg.multi_dot(Rn_inv, steer_vector,
                                    np.linalg.inv(np.linalg.multi_dot(steer_vector.T, Rn_inv, steer_vector)))
            # specs M x T x F
            self.enhanced_spec[:, f] = np.dot(w.T * self.stft_data[:, :, f])

        frames_enhanced = librosa.istft(self.enhanced_spec, hop_length=self.hop_length)


# def vad(audios):
#     frames=librosa.util.frame(np.ascontiguousarray(audios),frame_length=1024,hop_length=256)
#     spec=np.


if __name__ == '__main__':
    # data, sr = read_wav('./mvdr.wav')
    # stftdata = librosa.stft(data[:, 0])
    testsolution = CGMM_MVDR(path=r'./mvdr.wav')
    testsolution.train_CGMM()
    pass
