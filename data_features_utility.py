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
        audio_data, sampleRate = sf.read(audio_path)
        print('sample rate :{0}'.format(sampleRate))
        print('shape: {0}'.format(audio_data.shape))
        return audio_data

    def feature_extract(self, audio_path):
        audio_data = self.read_wav(audio_path)

        chan_0 = audio_data[:, 0]
        chan_1 = audio_data[:, 1]
        chan_2 = audio_data[:, 2]
        chan_3 = audio_data[:, 3]


        pass

    def hz2mel(hz):
        """Convert frequency to Mel frequency.
        Args:
            hz: Frequency.
        Returns:
            Mel frequency.
        """
        return 2595 * np.log10(1 + hz / 700.0)

    def mel2hz(mel):
        """Convert Mel frequency to frequency.
        Args:
            mel:Mel frequency
        Returns:
            Frequency.
        """
        return 700 * (10 ** (mel / 2595.0) - 1)

