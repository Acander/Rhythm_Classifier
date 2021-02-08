# This class file contains functions for retrieving the waveforms from the mp3 files and generates a dataset.
# The dataset used is the extended ballroom dataset

import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
import os

data_set_folder = "C:/Users/adria/PycharmProjects/Rhythm_Classifier/extendedballroom_v1.1/"

folder_names = [
    "Chacha",
    "Foxtrot",
    "Jive",
    "Pasodoble",
    "Quickstep",
    "Rumba",
    "Salsa",
    "Samba",
    "Slowwaltz",
    "Tango",
    "Viennesewaltz",
    "Waltz",
    "Wcswing"
]


def plot_waveform(samples, sample_rate, title):
    # The following makes the plot look nice
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    x = np.arange(len(samples)) / sample_rate
    plt.plot(x, samples)
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    plt.title(title)
    plt.show()


def load_sample(filename, time_s=None):
    y, sample_rate = lib.load(filename)
    if time_s:
        n_samples = int(sample_rate * time_s)
        samples = y.data[0:n_samples]
    else:
        samples = y.data
    plot_waveform(samples, sample_rate, "Waveform")
    return samples, sample_rate


def load_samples(class_size=None, n_classes=len(folder_names)):
    dataset = []
    for i in range(n_classes):
        print("Loading class: " + str(i))
        dance_class = []
        for j, filename in enumerate(os.listdir(data_set_folder + folder_names[i])):
            if class_size is not None:
                if j > class_size:
                    break
            music_sample, _ = load_sample(data_set_folder + folder_names[i] + "/" + filename)
            dance_class.append(np.array(music_sample))
        dataset.append(dance_class)

    print(np.array(dataset))
    return dataset


def test_load():
    test_file = "C:/Users/adria/PycharmProjects/Rhythm_Classifier/extendedballroom_v1.1/Chacha/100701.mp3"
    samples, sample_rate = load_sample(test_file)
    plot_waveform(samples, sample_rate, "Chacha")


if __name__ == '__main__':
    dataset = load_samples(class_size=4, n_classes=2)
