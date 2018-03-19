import numpy as np
import librosa
import librosa.display as rosa_display
import matplotlib.pyplot as plt
import os
import glob
import sys
import tensorflow as tf

class feature_utility():
    def __init__(self):
        # self.dir_Path=dir_Path
        pass

    def save_features_TFRecord(self,TFRecord_dir_path):
        if not os.path.exists(TFRecord_dir_path):
            os.mkdir(TFRecord_dir_path)

    def feature_extract(self, audio_path,n_mfcc_parm=40):
        # librosa will convert sample rate to 22050 by default
        # switch to soundfile if needed
        raw_audio, sampleRate = librosa.load(audio_path)
        mfcc=librosa.feature.mfcc(y=raw_audio,sr=sampleRate,n_mfcc=n_mfcc_parm)
        return mfcc

    def load_audio_files(self,dir_path):
        audio_pathes=os.scandir(dir_path)

        classDirNames=list(filter(lambda x:x.is_dir()==True,audio_pathes))

        for dirName in classDirNames:
            self.feature_extract(dir)



