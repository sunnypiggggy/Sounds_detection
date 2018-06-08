import matlab.engine as m_eng
import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display as rdis

import numpy as np
import scipy as sci
from multiprocessing import Process, Pool

import pickle
import gzip
from tqdm import tqdm


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

    def pre_emphasis(x)
        b = np.asarray([1.0, -0.97])
        emphasized_sig = sci.signal.lfilter(b=b, a=1.0, x=x)
        return emphasized_sig

    for var in tqdm(audio_dir, desc='process {0}'.format(process_i)):
        audio_data, sr = read_wav(var)

        acr_stft = []
        t = []
        for i in range(audio_data.shape[1]):
            mean = np.mean(audio_data[:, i])
            std = np.std(audio_data[:, i])
            x = (audio_data[:, i] - mean) / std
            t.append(x)
            x=pre_emphasis(x)

            tt=librosa.core.stft(x,n_fft=1024,hop_length=512)
            # temp = librosa.amplitude_to_db(
            #     librosa.core.stft(
            #         librosa.core.autocorrelate(x),
            #         n_fft=1024,
            #         hop_length=512))

            # acr_stft.append(temp)
        t = np.asarray(t)
        acr_stft = np.asarray(acr_stft)

        feature_dict = {
            'acr_stft': acr_stft,
            'shape': acr_stft.shape
        }
        save_name = var.split('\\')[-1].split('.wav')[0] + '.gzip'
        with gzip.open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(feature_dict, f)


if __name__ == '__main__':
    # eng = m_eng.start_matlab()

    dataset_dir = "DCASE2018-task5-dev"
    audio_dirs = list(os.scandir(os.path.join(dataset_dir, 'audio')))
    feature_ACR_dirs = "ACR_stft"

    if not os.path.exists(feature_ACR_dirs):
        os.mkdir(feature_ACR_dirs)

    audio_path = [[] for _ in range(6)]
    for i in range(len(audio_dirs)):
        if audio_dirs[i].name.split('.')[1] == 'wav':
            audio_path[i % 6].append(audio_dirs[i].path)

    # process_pool = Pool()
    # for i in range(6):
    #     process_pool.apply_async(worker, args=(audio_path[i], feature_ACR_dirs,i))
    #
    # process_pool.close()
    # process_pool.join()
    worker(audio_path[0], feature_ACR_dirs, 0)
    print("main process ends")

    # feature_acr_stft = list(os.scandir(feature_ACR_dirs))
    # with gzip.open(feature_acr_stft[0], 'rb') as f:
    #     xx = pickle.load(f)
    #     pass

    pass
