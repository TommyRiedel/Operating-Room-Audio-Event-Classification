import random
import librosa.effects
import numpy as np


# adding white noise
def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise_factor * noise
    return augmented_signal

# time stretch
def time_stretch(signal, stretch_rate):
    return librosa.effects.time_stretch(signal, stretch_rate)

# pitch scaling
def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr, num_semitones)

# polarity inversion
def invert_polarity(signal):
    return signal * -1

# random gain
def random_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = random.uniform(min_gain_factor, max_gain_factor)
    return signal * gain_factor