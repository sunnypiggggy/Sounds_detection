import matlab.engine as m_eng
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display as rdis
import soundfile as sf
import numpy as np
import scipy as sci

if __name__ == '__main__':

    # audio_raw,sampleRate = sf.read('absence.wav')

    eng=m_eng.start_matlab()

    # audio_data=eng.cgmm_mvdr('absence.wav',20)
    # audio_data=np.asarray(audio_data).squeeze()
    # audio_data=sci.signal.wiener(audio_data)
    # plt.plot(audio_data)
    # plt.show()
    # sf.write('./mvdr_out.wav',audio_data,16000)
    morse,bump,num_features,num_channel=eng.compute_wavelet_py('cooking.wav',nargout=4)
    # [a,b,c,d]=eng.compute_wavelet_py('cooking.wav',nargout=4)
    # tt=eng.shit()
    # a, b, c, d = eng.shit()
    morse=np.asarray(morse)
    bump=np.asarray(bump)
    num_features=int(num_features)
    num_channel=int(num_channel)
    # morse=morse.reshape([num_channel,num_features,-1])
    morse = morse.reshape([-1, num_features, num_channel]).transpose([2,1,0])
    bump=bump.reshape([-1, num_features, num_channel]).transpose([2,1,0])
    rdis.specshow(morse[0])
    plt.show()

    pass



