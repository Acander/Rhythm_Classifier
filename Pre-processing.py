import matplotlib.pyplot as plt
import numpy as np
import madmom as mm
import dataHandler as dh
import scipy.signal as sps


def plot_waveform(samples, title, sample_rate=0.0):
    # The following makes the plot look nice
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    x = np.arange(len(samples))
    if sample_rate:
        x = x / sample_rate
    plt.plot(x, samples)
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    plt.title(title)
    plt.show()


def load_samples_from_text(class_name):
    dataset = np.load(class_name + '.npy')
    return dataset


def gen_tempogram(sample):
    sample_len = len(sample) / dh.SAMPLE_RATE
    spec = mm.audio.spectrogram.Spectrogram(sample)
    sf = mm.features.onsets.spectral_diff(spec)
    sf_sample_rate = len(
        sf) / sample_len  # Divde the sample length (in seconds) by n points in the spectral flux to get spectral flux sample_rate. (Inverse)
    print(sf_sample_rate * sample_len)
    # plot_waveform(sf, sample_rate=sf_sample_rate, title="Spectral Flux")
    tempogram = fourier_transform(sf, sf_sample_rate)
    return tempogram


def fourier_transform(onsets, sample_rate):
    win_size = sample_rate * 8
    overlap = int(win_size - sample_rate * 0.5)
    freq, time, stft = sps.stft(onsets, fs=sample_rate, nperseg=win_size, noverlap=overlap,
                                nfft=8192, window='hann')
    tempo = hertz_to_tempo(freq)
    magnitudes = np.abs(stft)
    plot_tempogram(magnitudes, time, tempo)

    return magnitudes


def plot_tempogram(magnitudes, time, tempo):
    cmap = plt.get_cmap('copper')
    plt.ylim([25, 200])
    plt.pcolormesh(time, tempo, magnitudes, cmap=cmap)
    plt.set_cmap(cmap)
    plt.title('Tempogram')
    plt.ylabel('BPM')
    plt.xlabel('Time (s)')
    plt.show()


def hertz_to_tempo(frequencies):
    return 60 * frequencies  # Formula 6.24


if __name__ == '__main__':
    chachas = load_samples_from_text(dh.FOLDER_NAMES[0])
    foxtrots = load_samples_from_text(dh.FOLDER_NAMES[1])
    # plot_waveform(chachas[0], sample_rate=dh.SAMPLE_RATE, title=dh.FOLDER_NAMES[0])
    gen_tempogram(sample=chachas[0])
    gen_tempogram(sample=chachas[1])
    gen_tempogram(sample=chachas[2])
    gen_tempogram(sample=chachas[3])
    gen_tempogram(sample=foxtrots[0])
    gen_tempogram(sample=foxtrots[1])
    gen_tempogram(sample=foxtrots[2])
    gen_tempogram(sample=foxtrots[3])

    

