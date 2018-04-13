import librosa
import numpy as np
import pickle
import glob
import os

class data_feature():
    def __init__(self):
        self.datasets=[]
        self.bands=128
        self.frames=41

    def __window(self,data, windows_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + windows_size)
            start += (windows_size // 2)

    def feature_extract(self,audioFilePath, bands=128, frames=41):
        window_size = 512 * (frames - 1)
        mfccs = []
        # librosa will convert sample rate to 22050 by default
        # switch to soundfile if needed
        raw_audio, sampleRate = librosa.load(audioFilePath)
        for (start, end) in self.__window(raw_audio, window_size):
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

    def save_features(self,dir_path):
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
                features = self.feature_extract(fn, 256, 41)
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
