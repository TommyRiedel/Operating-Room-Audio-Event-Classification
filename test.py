import librosa
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import math


def freqToMel(f):
    return 2595 * math.log(1+(f/700), 10)

freqToMelv = np.vectorize(freqToMel)
Hz = np.linspace(0,1.2e4)
Mel = freqToMelv(Hz)


plt.plot(Hz, Mel)
plt.title('Hertz to Mel')
plt.xlabel('Hertz Scale')
plt.ylabel('Mel Scale')
plt.savefig('final_pictures/mel1.pdf')
plt.show()


sr = 44100
n_fft = 2048
n = 10
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n, fmin=0, fmax=12000)
mel_basis /= np.max(mel_basis, axis=-1)[:, None]
f = np.linspace(0, 22050, 1025)
f_all = np.matlib.repmat(f, n,1)
plt.figure()
for i in range(n):
    plt.plot(f,mel_basis[i])
plt.xlim([0, 12000])
plt.ylim([0, 1])
plt.xlabel('Frequency (Hz)')
plt.title('Mel Filter Bank (n_mels=10)')
plt.savefig('final_pictures/mel2.pdf')
plt.show()