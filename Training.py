from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import dataHandler as dh

# Here we use a KNN classifier for distinguishing the different rhythms.


TEST_PARTION = 0.2


def load_samples_from_text(class_name):
    dataset = np.load('OnSetPatterns/' + class_name + '.npy')
    return dataset


def get_train_data():
    x = []
    y = []
    for i, class_name in enumerate(dh.FOLDER_NAMES):
        onset_pattern_samples = load_samples_from_text(class_name)
        for sample in onset_pattern_samples:
            x.append(sample)
            y.append(i)
    return x, y


def train_model(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_train, y_train)
    return model


def evaluation(y_test, y_predict):
    print_confusion_matrix(y_test, y_predict)
    print("\n")
    print_classification_report(y_test, y_predict)


def print_confusion_matrix(y_test, y_predict):
    cm = confusion_matrix(y_test, y_predict)
    print("Number of Samples: ", len(y_test))
    print("Confusion Matrix: \n", cm)


def print_classification_report(y_test, y_predict):
    cr = classification_report(y_test, y_predict)
    print("Classification Report: \n", cr)


def training():
    x, y = get_train_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_PARTION)
    model = train_model(x_train, y_train)
    y_predict = model.predict(x_test)
    evaluation(y_test, y_predict)


if __name__ == '__main__':
    training()
