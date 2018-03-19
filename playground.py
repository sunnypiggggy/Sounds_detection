import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Fs=22050
# x=np.arange(Fs)
# y=np.sin(2*np.pi*10*x/Fs)+np.sin(2*np.pi*6000*x/Fs)+np.sin(2*np.pi*10000*x/Fs)
# plt.figure()
# plt.plot(y)
# plt.figure()

# mfcc = librosa.feature.mfcc(y=y, sr=Fs, n_mfcc=256)
# plt.imshow(mfcc)
# plt.show()
# librosa.display.specshow(mfcc, sr=Fs, x_axis='time')
# stft=librosa.stft(y)
# plt.figure()
# librosa.display.specshow(stft, sr=Fs, x_axis='time')


# y, sr = librosa.load('E:\\myDoc\\pyCharmWorkspace\\Sounds_detection\\ESC-50-master\\101 - Dog\\1-100032-A.ogg')
# sr=Fs
# mfcc=librosa.feature.mfcc(y,sr,n_mfcc=128,hop_length=512)
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfcc, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()


dataset=np.random.randint(10,size=10)
label=np.random.randint(2,size=10)
test={
    
}


pass
