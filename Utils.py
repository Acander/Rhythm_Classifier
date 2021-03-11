import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import matplotlib.ticker as ticker


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


def plot_fourier_spectrogram(magnitudes, x, y, title, x_label, y_label, lim=True):
    cmap = plt.get_cmap('copper')
    if lim:
        plt.ylim([0, 5000])
    plt.pcolormesh(x, y, magnitudes, cmap=cmap)
    plt.set_cmap(cmap)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def plot_spectrogram(magnitudes):
    cmap = plt.get_cmap('copper')
    plt.pcolormesh(magnitudes, cmap=cmap)
    plt.set_cmap(cmap)
    plt.show()


def plot_mel_spectrogram(log_S, sample_rate, hop_size):
    # Make a new figure
    plt.figure(figsize=(12, 4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sample_rate, hop_length=hop_size, x_axis='time', y_axis='mel')
    plt.ylim([0, 5000])

    # Put a descriptive title on the plot
    plt.title('Mel Spectrogram')
    cmap = plt.get_cmap('copper')
    plt.set_cmap(cmap)
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 1000))  # Transform into kHZ
    plt.gca().yaxis.set_major_formatter(ticks)
    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHZ)')

    # Make the figure layout compact
    plt.tight_layout()
    plt.show()
