# All parameters extracted from Holzapfel et. al, "IMPROVING TEMPO-SENSITIVE AND TEMPO-ROBUST DESCRIPTORS
# FOR RHYTHMIC SIMILARITY", OFAI, 2011

import numpy as np
import madmom as mm
import dataHandler as dh
import scipy.signal as sps
import Utils
import librosa

SAMPLE_TIME_SEC = 30

WIN_SIZE_1 = int(dh.SAMPLE_RATE * 0.00464)
HOP_SIZE_1 = int(WIN_SIZE_1 * 0.5)
N_BANDS_1 = 32
SUBBAND_MASK_WIN_SIZE = 11

COMPRESSION_1 = 1

WINL = 8  # Taken from original paper
HOP_SIZE_2 = 0.5  # Taken from original paper
NBINS = 5

COMPRESSION_2 = 8

def load_samples_from_text(class_name):
    dataset = np.load(class_name + '.npy')
    return dataset


def short_time_fourier_transform(waveform, sample_rate, win_size, hop_size):
    freq, time, stft = sps.stft(waveform, fs=sample_rate, nperseg=win_size, noverlap=win_size - hop_size,
                                nfft=8192, window='hann')
    magnitudes = np.abs(stft)
    return freq, time, magnitudes


def mel_spectrogram(sample):
    S = librosa.feature.melspectrogram(np.array(sample), sr=dh.SAMPLE_RATE, n_mels=85, n_fft=8192, window='hann',
                                       win_length=WIN_SIZE_1, hop_length=HOP_SIZE_1)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def filter_bank_processing(sample):
    freq, time, magnitudes = short_time_fourier_transform(np.array(sample), dh.SAMPLE_RATE, WIN_SIZE_1, HOP_SIZE_1)
    magnitudes = np.dot(mm.audio.filters.LogFilterbank(freq, num_bands=12).transpose(), magnitudes)
    compressed_spectrogram = compression(magnitudes, N_BANDS_1)
    log_S = librosa.power_to_db(np.array(compressed_spectrogram), ref=np.max)
    return log_S


def log_stft_1(sample):
    # return mel_spectrogram(sample)
    return filter_bank_processing(sample)


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


def compression(spectrogram, compression_size):
    equidistant_step = int(len(spectrogram) / compression_size)
    compressed_spectrogram = []
    for i in range(0, compression_size * equidistant_step, equidistant_step):
        compressed_spectrogram.append(spectrogram[-i])
    return compressed_spectrogram


def periodicity_spectrum(compressed_spectrogram):
    periodicity_spectrogram = []
    sample_rate = len(compressed_spectrogram[0]) / SAMPLE_TIME_SEC
    for band in compressed_spectrogram:
        freq, time, magnitudes = short_time_fourier_transform(band, sample_rate, int(WINL * sample_rate),
                                                              int(HOP_SIZE_2 * sample_rate))
        filter_bank = mm.audio.filters.LogFilterbank(freq, num_bands=5)
        filtered_spectrum = np.dot(filter_bank.transpose(), magnitudes)
        # Normalization
        periodicity_spectrogram.append(np.average(filtered_spectrum, axis=0))

    return periodicity_spectrogram

if __name__ == '__main__':
    chachas = load_samples_from_text(dh.FOLDER_NAMES[0])
    # Utils.plot_waveform(chachas[0], title="A simple chacha", sample_rate=dh.SAMPLE_RATE)
    # freq, time, stft = fourier_transform(chachas[0], sample_rate=dh.SAMPLE_RATE)
    # Utils.plot_spectrogram(librosa.amplitude_to_db(stft, ref=np.max), time, freq, "Fourier", "time", "frequency")
    log_S = log_stft_1(chachas[0])
    # Utils.plot_mel_spectrogram(log_S, dh.SAMPLE_RATE, HOP_SIZE)
    Utils.plot_spectrogram(log_S)
    masked_spectrogram = unsharp_mask(log_S)
    Utils.plot_spectrogram(masked_spectrogram)
    # compressed_spectrogram = compression(masked_spectrogram, 38)
    periodicity_spectrogram = periodicity_spectrum(masked_spectrogram)
    Utils.plot_spectrogram(periodicity_spectrogram)
    compressed_periodicity = compression(periodicity_spectrogram, COMPRESSION_2)
    Utils.plot_spectrogram(compressed_periodicity)
