# All parameters extracted from Holzapfel et. al, "IMPROVING TEMPO-SENSITIVE AND TEMPO-ROBUST DESCRIPTORS
# FOR RHYTHMIC SIMILARITY", OFAI, 2011

import numpy as np
import madmom as mm
import dataHandler as dh
import scipy.signal as sps
import Utils
import librosa


WIN_SIZE = int(dh.SAMPLE_RATE * 0.0464)
HOP_SIZE = int(WIN_SIZE * 0.5)
N_BANDS_1 = 85

SUBBAND_MASK_WIN_SIZE = 11


def load_samples_from_text(class_name):
    dataset = np.load(class_name + '.npy')
    return dataset


def fourier_transform(waveform, sample_rate):
    win_size = int(sample_rate * 0.0464)
    overlap = int(win_size * 0.5)
    freq, time, stft = sps.stft(waveform, fs=sample_rate, nperseg=win_size, noverlap=overlap,
                                nfft=8192, window='hann')
    magnitudes = np.abs(stft)
    return freq, time, magnitudes


def filter_bank_processing(sample):
    S = librosa.feature.melspectrogram(np.array(sample), sr=dh.SAMPLE_RATE, n_mels=85, n_fft=8192, window='hann',
                                       win_length=WIN_SIZE, hop_length=HOP_SIZE)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def unsharp_mask(spectrogram):
    # Each subband time step corresponds to approximatly 0.023ms of length -> 11 * 0.023 = 0.253
    masked_spectrogram = []
    for subband in spectrogram:
        masked_subband = []
        for time_step in range(0, len(subband), SUBBAND_MASK_WIN_SIZE):
            if len(subband) < time_step + SUBBAND_MASK_WIN_SIZE:
                interval_length = len(subband) - time_step
            else:
                interval_length = SUBBAND_MASK_WIN_SIZE
            masked_subband = concatenate_lists(masked_subband, mask(subband, time_step, interval_length))
        masked_spectrogram.append(masked_subband)
    return np.array(masked_spectrogram)


def mask(subband, time_step, interval_length):
    current_interval = subband[time_step:time_step + interval_length]
    average_magnitude = np.average(current_interval)
    return current_interval - average_magnitude


def concatenate_lists(list1, list2):
    return np.concatenate((list1, list2))


if __name__ == '__main__':
    chachas = load_samples_from_text(dh.FOLDER_NAMES[0])
    # Utils.plot_waveform(chachas[0], title="A simple chacha", sample_rate=dh.SAMPLE_RATE)
    # freq, time, stft = fourier_transform(chachas[0], sample_rate=dh.SAMPLE_RATE)
    # Utils.plot_spectrogram(librosa.amplitude_to_db(stft, ref=np.max), time, freq, "Fourier", "time", "frequency")
    log_S = filter_bank_processing(chachas[0])
    # Utils.plot_mel_spectrogram(log_S, dh.SAMPLE_RATE, HOP_SIZE)
    masked_spectrogram = unsharp_mask(log_S)
    Utils.plot_mel_spectrogram(np.array(masked_spectrogram), dh.SAMPLE_RATE, HOP_SIZE)
