from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import dataHandler as dh
import Utils
import OnsetPattern

# Here we use a KNN classifier for distinguishing the different rhythms.

"""
455
507
350
53
497
470
47
468
65
464
252
529
23
"""

"Chacha"
"Foxtrot"
"Jive"
"Pasodoble"
"Quickstep"
"Rumba"
"Salsa"
"Samba"
"Slowwaltz"
"Tango"
"Viennesewaltz"
"Waltz"
"Wcswing"


TEST_PARTION = 0.2
N_NEIGHBOURS = 20
MIN_LEN = 455  # 45 % accuracy for 7 classes


def load_samples_from_text(class_name):
    dataset = np.load('Onset_Patterns/' + class_name + '.npy')
    return dataset


def get_train_data():
    x = []
    y = []
    for i, class_name in enumerate(dh.FOLDER_NAMES):
        onset_pattern_samples = load_samples_from_text(class_name)
        if MIN_LEN <= len(onset_pattern_samples):
            print(class_name)
            onset_pattern_samples = onset_pattern_samples[0:MIN_LEN]
            #print(len(onset_pattern_samples))
            for j, sample in enumerate(onset_pattern_samples):
                x.append(sample)
                y.append(i)
                # if j == 0:
                #    Utils.plot_spectrogram(np.reshape(sample, OnsetPattern.ONSET_PATTERN_FINAL_SHAPE))
    return x, y


def check_data():
    for i, class_name in enumerate(dh.FOLDER_NAMES):
        onset_pattern_samples = load_samples_from_text(class_name)
        print(len(onset_pattern_samples[0]))

def train_model(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
    model.fit(x_train, y_train)
    return model


def evaluation(y_test, y_predict):
    print_confusion_matrix(y_test, y_predict)
    print("\n")
    print_classification_report(y_test, y_predict)
    print("\n")


def print_confusion_matrix(y_test, y_predict):
    cm = confusion_matrix(y_test, y_predict)
    print("Number of Samples: ", len(y_test))
    print("Confusion Matrix: \n", cm)


def print_classification_report(y_test, y_predict):
    cr = classification_report(y_test, y_predict)
    print("Classification Report: \n", cr)


def training():
    x, y = get_train_data()
    print(len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_PARTION, shuffle=True)
    model = train_model(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluation(y_test, y_predict)


if __name__ == '__main__':
    training()
    # check_data()
