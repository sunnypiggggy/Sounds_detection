import numpy as np
import librosa
import librosa.display as rosa_display
import matplotlib.pyplot as plt
import os
import glob
import sys
import pickle

bands = 128
frame = 41


def window(data, windows_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + windows_size)
        start += (windows_size // 2)


def feature_extract(audioFilePath, bands=128, frames=41):
    window_size = 512 * (frames - 1)
    mfccs = []
    # librosa will convert sample rate to 22050 by default
    # switch to soundfile if needed
    raw_audio, sampleRate = librosa.load(audioFilePath)
    for (start, end) in window(raw_audio, window_size):
        if (len(raw_audio[start:end]) == window_size):
            sound_clip = raw_audio[start:end]
            mfcc = librosa.feature.mfcc(y=sound_clip, sr=sampleRate, n_mfcc=bands)
            # mfcc= librosa.feature.melspectrogram(y=sound_clip,sr=sampleRate)
            # mfcc = librosa.logamplitude(mfcc)
            # mfcc = librosa.feature.mfcc(y=sound_clip, sr=sampleRate, n_mfcc=bands).T.flatten()[:, np.newaxis].T

            mfccs.append(mfcc)
    # features = np.asarray(mfccs).reshape(len(mfccs), frames, bands)
    features = np.asarray(mfccs)
    return features


def save_features(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    rootPath = os.getcwd()
    audioPaths = os.scandir(os.path.join(rootPath, 'ESC-50-master'))

    classDirNames = list(filter(lambda x: x.is_dir() == True, audioPaths))
    feature_list = []
    for dirName in classDirNames:
        audio_files = glob.glob(os.path.join(dirName.path, "*.ogg"))
        print(dirName.name)
        name_list = dirName.name.split(' - ')

        feature_list.clear()
        for fn in audio_files:
            print(fn)
            features = feature_extract(fn, 256, 41)
            feature_list.append(features)
            # plt.figure()
            # for i in range(len(features)):
            #     plt.subplot(1,len(features),i+1)
            #     plt.axis('off')
            #     plt.imshow(features[i])
            # plt.show()
        features_dict = {
            'label': int(name_list[0]),
            'image': feature_list,
            'name': name_list[1]
        }
        with open(os.path.join(dir_path, dirName.name), 'wb') as f:
            pickle.dump(features_dict, f)
        print('Dumped {0}'.format(dirName.name))


def load_features(dir_path):
    # rootPath = os.getcwd()
    # feature_path = os.scandir(os.path.join(rootPath, dir_path))
    data=[]
    try:
        feature_path = os.scandir(dir_path)
    except Exception as e:
        print(e)
        return None

    for file in feature_path:
        with open(file.path,'rb') as  file_handle:
            tt = pickle.load(file_handle)
            data.append(tt)
            print('loading '+tt['name'])
    return data


if __name__ == "__main__":
    # save_features('features')
    load_features('features')
