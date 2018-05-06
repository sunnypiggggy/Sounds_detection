import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf

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


# dataset=np.random.randint(10,size=10)
# label=np.random.randint(2,size=10)
# test={
#
# }
dir_name="DCASE2018-task5-dev"
dir_fold_name="evaluation_setup"
meta_date="meta.txt"
# full_path=os.path.join(dir_name,meta_date)
# print(full_path)
# try:
#     wav_files=[]
#     label_set=set([])
#     with open(full_path,'r+') as f_meta:
#         # print(f_meta.readline())
#         for line in f_meta.readlines():
#             print(line.split())
#             label_set.add(line.split()[1])
#     label_set=sorted(label_set)
#     i=0
#     with open(os.path.join( dir_name,'label_set.txt'),'w+') as f:
#         for line in list(label_set):
#             # f.write(" '"+line+"' "+":"+str(i)+" ,")
#             f.write("'"+line+"', ")
#             i+=1
#
#
#
# except Exception as e:
#     print(e)

# test_audio=os.path.join(dir_name,'audio','DevNode1_ex1_1.wav')
# print(test_audio)
# audio_date,sampleRate=sf.read(test_audio)
#
#
# pass
