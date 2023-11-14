"""
Data Augmentation of the mel-power spectrograms.
Related paper : https://arxiv.org/pdf/1904.08779.pdf

In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import librosa
import librosa.display
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
import numpy as np
import random
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def spec_augment(
    mel_spectrogram,
    frequency_masking_para=27,
    time_masking_para=100,
    frequency_mask_num=1,
    time_mask_num=1,
):
    """Spec augmentation Calculation Function.
    First step is frequency masking, second step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F", default = 100
      time_masking_para(float): Augmentation parameter, "time mask parameter T", default = 27
      frequency_mask_num(float): number of frequency masking lines, "m_F", default = 1
      time_mask_num(float): number of time masking lines, "m_T", default = 1

    # Returns
      mel_spectrogram(numpy array): masked mel spectrogram.
    """
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    # Step 1 : Frequency masking = Setting random number of consecutive frequency bands to zero
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[f0 : f0 + f, :] = 0

    # Step 2 : Time masking =  Setting a random number of consecutive time frames to zero
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[:, t0 : t0 + t] = 0

    return mel_spectrogram
