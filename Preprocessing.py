""" Script for the preprocessing of the audio snippets into spectrograms together with data augmentation (of Audio and Spectrograms):
"""
import os
import librosa.display
import numpy as np
import math
import matplotlib.pyplot as plt
from Augment_spec import spec_augment
import tensorflow.compat.v1 as tf
from Augment_wave import add_white_noise, random_gain

tf.disable_v2_behavior()

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# create subfolder
if not os.path.exists("spectrograms/"):
    os.makedirs("spectrograms/")

# Number of mel filter banks = 64
number_mels = 64
clrmap = "magma"
window_length_s = 0.7
save_path = "spectrograms/spectrograms"

classes = os.listdir("wavfiles")
if ".DS_Store" in classes:
    classes.remove(".DS_Store")

for f in classes:
    # count_f += 1

    # Creating folders for storing the (augmented) spectrograms
    if not os.path.exists(save_path + "_" + f + "/"):
        os.makedirs(save_path + "_" + f + "/")

    if not os.path.exists(save_path + "_" + f + "_Augmented/"):
        os.makedirs(save_path + "_" + f + "_Augmented/")

    if not os.path.exists(save_path + "_" + f + "_Augmented_wav/"):
        os.makedirs(save_path + "_" + f + "_Augmented_wav/")

    data = os.listdir("wavfiles" + "/" + f)
    if ".DS_Store" in data:
        data.remove(".DS_Store")

    # Iterating through the different classes (f): Loading + Augmenting + Windowing + generates Mel power spectrogram + Augmenting + Saves spectrograms
    for g in data:
        # Loads the wavefiles
        y, sr = librosa.load("wavfiles" + "/" + f + "/" + g, sr=None, mono=True)

        # Adding white noise or random gain -> Generates augmented version of audio
        y_aug_1 = add_white_noise(y, 0.1)
        y_aug_2 = add_white_noise(y, 0.25)
        y_aug_3 = random_gain(y, 2, 4)

        # Window length absed on the samplerate and window length in seconds
        window_length = int(sr * window_length_s)
        i = 0
        count_file = 0

        while i <= (len(y) - window_length):
            # 0.7 second long wavefiles
            window = y[i : i + window_length]
            # augmented wavefiles (WaveAug)
            window_aug_1 = y_aug_1[i : i + window_length]
            window_aug_2 = y_aug_2[i : i + window_length]
            window_aug_3 = y_aug_3[i : i + window_length]

            i += math.floor(window_length)
            count_file += 1

            # Mel-power spectrogram generation (Log spectrum) - maximum frequency = 12000 Hz - + visualization
            ms = librosa.feature.melspectrogram(
                window, sr=sr, n_mels=number_mels, fmax=12000, window="hamm"
            )
            S1_log = librosa.power_to_db(ms, ref=np.max)

            fig = plt.figure(0, frameon=False)
            fig.set_size_inches(3.80, 3.80)

            librosa.display.specshow(
                S1_log, cmap=clrmap, sr=sr, x_axis="time", y_axis="mel", fmax=12000
            )
            plt.axis("off")

            fig.savefig(
                save_path + "_" + f + "/" + g + "_" + str(count_file) + ".jpg",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )

            ## Similar steps for the augmented audio signals
            ms_aug1 = librosa.feature.melspectrogram(
                window_aug_1, sr=sr, n_mels=number_mels, fmax=12000, window="hamm"
            )
            S1_log_aug_1 = librosa.power_to_db(ms_aug1, ref=np.max)

            fig = plt.figure(0, frameon=False)
            fig.set_size_inches(3.80, 3.80)

            librosa.display.specshow(
                S1_log_aug_1,
                cmap=clrmap,
                sr=sr,
                x_axis="time",
                y_axis="mel",
                fmax=12000,
            )
            plt.axis("off")

            fig.savefig(
                save_path
                + "_"
                + f
                + "_Augmented_wav/"
                + g
                + "_"
                + str(count_file)
                + "_4"
                + ".jpg",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )

            ms_aug_2 = librosa.feature.melspectrogram(
                window_aug_2, sr=sr, n_mels=number_mels, fmax=12000, window="hamm"
            )
            S1_log_aug_2 = librosa.power_to_db(ms_aug_2, ref=np.max)

            fig = plt.figure(0, frameon=False)
            fig.set_size_inches(3.80, 3.80)

            librosa.display.specshow(
                S1_log_aug_2,
                cmap=clrmap,
                sr=sr,
                x_axis="time",
                y_axis="mel",
                fmax=12000,
            )
            plt.axis("off")

            fig.savefig(
                save_path
                + "_"
                + f
                + "_Augmented_wav/"
                + g
                + "_"
                + str(count_file)
                + "_2"
                + ".jpg",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )

            ms_aug_3 = librosa.feature.melspectrogram(
                window_aug_3, sr=sr, n_mels=number_mels, fmax=12000, window="hamm"
            )
            S1_log_aug_3 = librosa.power_to_db(ms_aug_3, ref=np.max)

            fig = plt.figure(0, frameon=False)
            fig.set_size_inches(3.80, 3.80)

            librosa.display.specshow(
                S1_log_aug_3,
                cmap=clrmap,
                sr=sr,
                x_axis="time",
                y_axis="mel",
                fmax=12000,
            )
            plt.axis("off")

            fig.savefig(
                save_path
                + "_"
                + f
                + "_Augmented_wav/"
                + g
                + "_"
                + str(count_file)
                + "_3"
                + ".jpg",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )

            # Augmenting the spectrograms! (SpecAug)
            # for the different classes different amount of data is available - to have the same amount of data for all classes (700/150/150 samples)
            # the data of different classes need to be expanded by different amounts of augmented data - to avoid memory problems the following defines
            # the amount of augmented data requried for each class

            if f == "011_mono_Koagulation" or f == "080_Idle":
                a = 4
            if (
                f == "012_bi_Koagulation"
                or f == "031_Haemolog_schnell"
                or f == "051_Tisch_hoch"
                or f == "052_Tisch_runter"
                or f == "053_Tisch_neigen"
                or f == "054_Tisch_vor_zurueck"
                or f == "070_DaVinci"
            ):
                a = 4
            if f == "060_Telefon":
                a = 16

            # the parameters to augmented the spectrograms (frequency + time masking)
            for j in range(0, a):
                if divmod(j, 4)[1] == 0:
                    frequency_masking_para = 6.75
                    time_masking_para = 25
                    frequency_masking_num = 1
                    time_masking_num = 1
                if divmod(j, 4)[1] == 1:
                    frequency_masking_para = 6.75
                    time_masking_para = 25
                    frequency_masking_num = 2
                    time_masking_num = 2
                if divmod(j, 4)[1] == 2:
                    frequency_masking_para = 3.75
                    time_masking_para = 17.5
                    frequency_masking_num = 2
                    time_masking_num = 2
                if divmod(j, 4)[1] == 3:
                    frequency_masking_para = 6.75
                    time_masking_para = 17.5
                    frequency_masking_num = 2
                    time_masking_num = 2
                ms = librosa.feature.melspectrogram(
                    window, sr=sr, n_mels=number_mels, fmax=12000, window="hamm"
                )
                ws = Augment_spec.spec_augment(
                    mel_spectrogram=ms,
                    frequency_masking_para=frequency_masking_para,
                    time_masking_para=time_masking_para,
                    frequency_mask_num=frequency_masking_num,
                    time_mask_num=time_masking_num,
                )
                # creating + saving of the augmented spectrograms
                fig.clf()
                fig1 = plt.figure(frameon=False, figsize=(3.80, 3.80))
                librosa.display.specshow(
                    librosa.power_to_db(ws, ref=np.max),
                    cmap=clrmap,
                    sr=sr,
                    x_axis="time",
                    y_axis="mel",
                    fmax=12000,
                )
                plt.axis("off")
                fig1.savefig(
                    save_path
                    + "_"
                    + f
                    + "_Augmented/"
                    + g
                    + "_"
                    + str(count_file)
                    + "_"
                    + str(j)
                    + ".jpg",
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                fig1.clf()
