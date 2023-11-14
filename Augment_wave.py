"""Augmentation of the wave-files:

1. Adding white noise
2. Time stretch
3. Pitch scaling
4. Polarity inversion
5. Random gain
"""

import random
import librosa.effects
import numpy as np


# Adding white noise (Random normal distribution between 0 and the signal's standard deviation)
def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise_factor * noise
    return augmented_signal


# Time stretch (Stretches or compresses the input signal's timeline according to the specified stretch ratio)
def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(signal, stretch_rate)


# Pitch scaling (Changes the pitch of the input signal by the specified number of semitones)
def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr, num_semitones)


# Polarity inversion (Reverses the polarity of the signal by multiplying the values with -1)
def invert_polarity(signal):
    return signal * -1


# Random gain (Changes the level of the input signal by multiplying it by a random gain factor that is between the specified minimum and maximum values)
def random_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor
