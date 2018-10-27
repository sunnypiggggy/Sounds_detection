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



def pre_emphasis(x):
    b = np.asarray([1.0, -0.97])
    emphasized_sig = sci.signal.lfilter(b=b, a=1.0, x=x)
    return emphasized_sig


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

if __name__ == '__main__':
    audio_data, sampleRate=read_wav('cooking.wav')
    sample=audio_data[:,0]
    plt.subplot(2,1,1)
    plt.plot(sample)
    plt.tick_params(axis='x',which='both',labelbottom=False,labelleft=False)
    plt.tick_params(axis='y',which='both',labelbottom=False,labelleft=False)
    plt.title('origin signal')

    sample_em=pre_emphasis(sample)
    plt.subplot(2,1,2)
    plt.plot(sample_em)
    plt.tick_params(axis='x',which='both',labelbottom=False,labelleft=False)
    plt.tick_params(axis='y',which='both',labelbottom=False,labelleft=False)
    plt.title('pre-emphasised signal')
    plt.show()
    print('shit')