import matlab.engine as m_eng
import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display as rdis

import numpy as np
import scipy as sci
# import scipy.fftpack as fftpack
from sklearn.decomposition import IncrementalPCA

from multiprocessing import Process, Pool

import pickle
import gzip
from tqdm import tqdm

import config as cfg

import sidekit

def worker(audio_dir: list, save_dir, class_index, process_i=0):
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

    def pre_emphasis(x):
        b = np.asarray([1.0, -0.97])
        emphasized_sig = sci.signal.lfilter(b=b, a=1.0, x=x)
        return emphasized_sig

    def get_next_batch(input, batch_size=10):
        start= 0
        while start<len(input):
            yield input[start:start + batch_size]
            start = start + 5

    pca = IncrementalPCA(n_components=900, batch_size=10)

    for var in tqdm(get_next_batch(audio_dir), desc='process {0}'.format(process_i)):
        audio_datas=[]
        for x in var:
            audio_data, sr = read_wav(x)
            audio_datas.append(audio_data[:, 0])

        features = []
        t = []
        for i in range(len(audio_datas)):
            mean = np.mean(audio_datas[i])
            std = np.std(audio_datas[i])
            x = (audio_datas[i] - mean) / std
            t.append(x)
            x = pre_emphasis(x)
            features.append(np.fft.fft(x,n=1024))
        # mean = np.mean(audio_data)
        # std = np.std(audio_data)
        # x = (audio_data - mean) / std
        # x = pre_emphasis(x)
        # features = np.fft.fft(x, n=1024)
        features = np.abs(np.asarray(features))

        pca.partial_fit(features)

        feature_dict = {
            'pca_model': pca,
            'class_index': class_index
        }
        save_name = 'class_' + str(class_index) + '.gzip'
        with gzip.open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(feature_dict, f)


if __name__ == '__main__':
    # eng = m_eng.start_matlab()

    dataset_dir = "DCASE2018-task5-dev"
    audio_dirs = list(os.scandir(os.path.join(dataset_dir, 'audio')))
    data_meta_dirs = list(os.scandir(os.path.join(dataset_dir, 'evaluation_setup')))
    feature_dirs = "SPCC"

    if not os.path.exists(feature_dirs):
        os.mkdir(feature_dirs)

    audio_class_dirs = {}
    for i in range(cfg.num_class):
        audio_class_dirs[i] = []

    for meta in data_meta_dirs:
        with open(meta, 'r') as f_meta:
            f_meta.seek(0)
            for audio_file_line in f_meta.readlines():
                path, label, sess_label = audio_file_line.split()
                file_name = path.split('/')[-1]
                audio_class_dirs[cfg.class_name2index[label]].append(os.path.join(dataset_dir, 'audio', file_name))

    # process_pool = Pool()
    # for i in range(6):
    #     process_pool.apply_async(worker, args=(audio_path[i], feature_dirs,i))
    #
    # process_pool.close()
    # process_pool.join()
    worker(audio_class_dirs[1], feature_dirs, 0, 0)
    print("main process ends")

    # feature_acr_stft = list(os.scandir(feature_ACR_dirs))
    # with gzip.open(feature_acr_stft[0], 'rb') as f:
    #     xx = pickle.load(f)
    #     pass

    pass
