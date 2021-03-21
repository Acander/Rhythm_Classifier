# All parameters extracted from Holzapfel et. al, "IMPROVING TEMPO-SENSITIVE AND TEMPO-ROBUST DESCRIPTORS
# FOR RHYTHMIC SIMILARITY", OFAI, 2011

import numpy as np
import madmom as mm
import dataHandler as dh
import scipy.signal as sps
import Utils
import librosa


###################################################

min_sample_size = 661248
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


ONSET_PATTERN_FINAL_SHAPE = (8, 62)


###################################################


def load_samples_from_text(class_name):
    dataset = np.load('Raw_Waveforms/' + class_name + '.npy', allow_pickle=True)
    return dataset


def load_onset_patterns_from_text(class_name):
    dataset = np.load('Onset_Patterns/' + class_name + '.npy', allow_pickle=True)
    return dataset


def save_onsets_to_text(dataset, class_name):
    np.save('Onset_Patterns/' + class_name, dataset)


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


def plot_spectrogram(log_S, masked_spectrogram, periodicity_spectrogram, compressed_periodicity):
    Utils.plot_spectrogram(log_S)
    Utils.plot_spectrogram(masked_spectrogram)
    Utils.plot_spectrogram(periodicity_spectrogram)
    Utils.plot_spectrogram(compressed_periodicity)


def generate_onset_patterns_test():
    chachas = load_samples_from_text(dh.FOLDER_NAMES[0])
    log_S = log_stft_1(chachas[0])
    Utils.plot_spectrogram(log_S)
    masked_spectrogram = unsharp_mask(log_S)
    periodicity_spectrogram = periodicity_spectrum(masked_spectrogram)
    compressed_periodicity = compression(periodicity_spectrogram, COMPRESSION_2)
    plot_spectrogram(log_S, masked_spectrogram, periodicity_spectrogram, compressed_periodicity)


def generate_onset_patterns():
    for class_name in dh.FOLDER_NAMES:
        print(class_name)
        class_periodicities = []
        samples = load_samples_from_text(class_name)
        for i, sample in enumerate(samples):
            sample = sample[0:min_sample_size]
            print("Currently computing sample: ", i, "/", len(samples))
            log_S = log_stft_1(sample)
            masked_spectrogram = unsharp_mask(log_S)
            periodicity_spectrogram = periodicity_spectrum(masked_spectrogram)
            compressed_periodicity = compression(periodicity_spectrogram, COMPRESSION_2)
            class_periodicities.append(np.array(compressed_periodicity).flatten())
        save_onsets_to_text(np.array(class_periodicities), class_name)


def load_test():
    onset_patterns = load_onset_patterns_from_text(dh.FOLDER_NAMES[0])
    for i, onset_pattern in enumerate(onset_patterns):
        print(np.shape(onset_pattern))
        Utils.plot_spectrogram(np.reshape(onset_pattern, ONSET_PATTERN_FINAL_SHAPE))


def find_shortest_piece():
    shortest_piece_len = np.Infinity
    for class_name in dh.FOLDER_NAMES:
        samples = load_samples_from_text(class_name)
        for i, sample in enumerate(samples):
            if len(sample) < shortest_piece_len:
                shortest_piece_len = len(sample)
    print(shortest_piece_len)


if __name__ == '__main__':
    # generate_onset_patterns_test()
    # generate_onset_patterns()
    find_shortest_piece()
    # load_test()
