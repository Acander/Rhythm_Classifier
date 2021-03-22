# This class file contains functions for retrieving the waveforms from the mp3 files.
# The dataset used is the extended ballroom dataset

import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
import os

DATA_SET_HANDLER = "C:/Users/adria/PycharmProjects/Rhythm_Classifier/extendedballroom_v1.1/"

FOLDER_NAMES = [
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

SAMPLE_RATE = 22050


def load_sample(filename, time_s=None):
    y, sample_rate = lib.load(filename)
    if time_s:
        n_samples = int(sample_rate * time_s)
        samples = y.data[0:n_samples]
    else:
        samples = y.data
    print(sample_rate)
    return samples, sample_rate


def load_samples(class_size=None, n_classes=len(FOLDER_NAMES)):
    for i in range(n_classes):
        print("Loading class: " + str(i))
        dance_class = []
        for j, filename in enumerate(os.listdir(DATA_SET_HANDLER + FOLDER_NAMES[i])):
            if class_size is not None:
                if j > class_size:
                    break
            music_sample, _ = load_sample(DATA_SET_HANDLER + FOLDER_NAMES[i] + "/" + filename)
            dance_class.append(np.array(music_sample))
        save_samples_to_text(np.array(dance_class), FOLDER_NAMES[i])
        # dataset.append(np.array(dance_class))


def save_samples_to_text(dataset, class_name):
    print(dataset)
    np.save(class_name, dataset)


def test_load():
    test_file = "C:/Users/adria/PycharmProjects/Rhythm_Classifier/extendedballroom_v1.1/Chacha/100701.mp3"
    samples, sample_rate = load_sample(test_file)
    print(sample_rate)


if __name__ == '__main__':
    # dataset = load_samples(class_size=4, n_classes=len(FOLDER_NAMES))
    dataset = load_samples()
