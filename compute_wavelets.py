import matlab.engine as m_eng
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display as rdis

import numpy as np
import scipy as sci
from multiprocessing import Process, Pool

import pickle
import gzip
from tqdm import tqdm


def worker(audio_dir: list, save_dir,process_i):
    eng = m_eng.start_matlab()

    for var in tqdm(audio_dir,desc='process {0}'.format(process_i)):
        morse, bump, num_features, num_channel = eng.compute_wavelet_py(var, nargout=4)
        morse = np.asarray(morse)
        bump = np.asarray(bump)
        num_features = int(num_features)
        num_channel = int(num_channel)

        morse = morse.reshape([-1, num_features, num_channel]).transpose([2, 1, 0])
        bump = bump.reshape([-1, num_features, num_channel]).transpose([2, 1, 0])

        feature_dict = {
            'morse': morse,
            'bump': bump,
            'shape': morse.shape
        }
        save_name = var.split('\\')[-1].split('.wav')[0] + '.gzip'
        with gzip.open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(feature_dict, f)

    pass


if __name__ == '__main__':
    # eng = m_eng.start_matlab()

    dataset_dir = "DCASE2018-task5-dev"
    audio_dirs = list(os.scandir(os.path.join(dataset_dir, 'audio')))
    feature_wavelets_dirs = "wavelets"

    if not os.path.exists(feature_wavelets_dirs):
        os.mkdir(feature_wavelets_dirs)

    # audio_path = [[] for _ in range(6)]
    # # m_engs=[m_eng.start_matlab() for _ in range(6)]
    #
    # for i in range(len(audio_dirs)):
    #     if audio_dirs[i].name.split('.')[1] == 'wav':
    #         audio_path[i % 6].append(audio_dirs[i].path)
    #    # for i in range(len(audio_dirs)):
    #     if audio_dirs[i].name.split('.')[1] == 'wav':
    #         audio_path[i % 6].append(audio_dirs[i].path)
    #
    # pass
    # process_pool = Pool()
    # for i in range(6):
    #     process_pool.apply_async(worker, args=(audio_path[i], feature_wavelets_dirs,i))
    #
    # process_pool.close()
    # process_pool.join()

    # audio_path =[]
    # for var in audio_dirs:
    #     if var.name.split('.')[1]=='wav':
    #         audio_path.append(var.path)
    # worker(audio_path, feature_wavelets_dirs,0)
    print("main process ends")

    feature_wavelets=list(os.scandir(feature_wavelets_dirs))
    with gzip.open(feature_wavelets[0],'rb') as f:
        xx=pickle.load(f)
        pass

    pass
