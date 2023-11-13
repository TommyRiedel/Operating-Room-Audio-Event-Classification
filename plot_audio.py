import librosa
import librosa.display
from librosa.display import waveshow
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


y_1, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/011_mono_Koagulation/001_011_02_00142.wav', duration=0.7)
y_2, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/012_bi_Koagulation/001_012_02_00001.wav', duration=0.7)
y_3, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/020_Cutting/002_020_03_00410.wav', duration=0.7)
y_4, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/031_Haemolog_schnell/001_031_02_00010.wav', duration=0.7)
y_5, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/051_Tisch_hoch/004_051_04_00533.wav', duration=0.7)
y_6, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/052_Tisch_runter/004_052_04_00542.wav', duration=0.7)
y_7, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/053_Tisch_neigen/004_053_04_00573.wav', duration=0.7)
y_8, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/054_Tisch_vor_zurueck/004_054_04_00589.wav', duration=0.7)
y_9, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/060_Telefon/001_060_02_00009.wav', duration=0.7)
y_10, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/074_DaVinci_weitere/002_074_03_00154.wav', duration=0.7)
y_11, sr = librosa.load('/Users/thomasriedel/Documents/SA_Programming/wavfiles_x/080_Idle/013_080_05_03189.wav', duration=0.7)


y_1 = np.array([(y_1 / np.max(np.abs(y_1)))], np.float32)
y_2 = np.array([(y_2 / np.max(np.abs(y_2)))], np.float32)
y_3 = np.array([(y_3 / np.max(np.abs(y_3)))], np.float32)
y_4 = np.array([(y_4 / np.max(np.abs(y_4)))], np.float32)
y_5 = np.array([(y_5 / np.max(np.abs(y_5)))], np.float32)
y_6 = np.array([(y_6 / np.max(np.abs(y_6)))], np.float32)
y_7 = np.array([(y_7 / np.max(np.abs(y_7)))], np.float32)
y_8 = np.array([(y_8 / np.max(np.abs(y_8)))], np.float32)
y_9 = np.array([(y_9 / np.max(np.abs(y_9)))], np.float32)
y_10 = np.array([(y_10 / np.max(np.abs(y_10)))], np.float32)
y_11 = np.array([(y_11 / np.max(np.abs(y_11)))], np.float32)


plt.figure(figsize=(25, 20))
plt.rcParams.update({'font.size': 18})
plt.subplots_adjust(hspace=0.4)
plt.subplot(6, 2, 1)
librosa.display.waveshow(y_1, sr=sr)
plt.title('Coagulation monopolar', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 2)
librosa.display.waveshow(y_2, sr=sr)
plt.title('Coagulation bipolar', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 3)
librosa.display.waveshow(y_3, sr=sr)
plt.title('Cutting', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 4)
librosa.display.waveshow(y_4, sr=sr)
plt.title('Hämolock', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 5)
librosa.display.waveshow(y_5, sr=sr)
plt.title('Table up', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 6)
librosa.display.waveshow(y_6, sr=sr)
plt.title('Table down', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 7)
librosa.display.waveshow(y_7, sr=sr)
plt.title('Table tilt', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 8)
librosa.display.waveshow(y_8, sr=sr)
plt.title('Table forth', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 9)
librosa.display.waveshow(y_9, sr=sr)
plt.title('Phone', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 10)
librosa.display.waveshow(y_10, sr=sr)
plt.title('Da-Vinci', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.subplot(6, 2, 11)
librosa.display.waveshow(y_11, sr=sr)
plt.title('Idle', loc='left', fontweight="bold")
plt.ylabel('Amplitude')
plt.savefig('final_pictures/waveshow.png')
plt.show()




