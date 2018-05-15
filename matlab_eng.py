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

    audio_data=eng.cgmm_mvdr('absence.wav',20)
    audio_data=np.asarray(audio_data).squeeze()
    audio_data=sci.signal.wiener(audio_data)
    plt.plot(audio_data)
    plt.show()
    sf.write('./mvdr_out.wav',audio_data,16000)


    pass



