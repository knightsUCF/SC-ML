from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np





def convert_labels_to_numeric(data):
    """
    numeric label key: (can be improved later to auto generate key based on different data)

    B cell - 0
    T cell - 1
    Granulocyte - 2
    NK cell - 3
    Monocyte - 4
    Unknown - 5
    """
    labels = []
    string_labels = data

    for label in string_labels:
        if label == 'B cell':
            labels.append(0)
            continue
        if label == 'T cell':
            labels.append(1)
            continue
        if label == 'Granulocyte':
            labels.append(2)
            continue
        if label == 'NK cell':
            labels.append(3)
            continue
        if label == 'Monocyte':
            labels.append(4)
            continue
        else:
            labels.append(5)
    return labels


def run_stats():
    global sample_size
    number_of_correct_guesses = 0
    number_of_incorrect_guesses = 0
    i = 0

    for data_point in range(sample_size):
        sample = np.asarray(x_verification[i])
        predicted_sample = clf.predict([sample])
        print(predicted_sample[0])
        print(y_verification[i])
        if predicted_sample[0] == y_verification[i]:
            number_of_correct_guesses += 1
        else:
            number_of_incorrect_guesses += 1
        i += 1

    print('number of correct guesses: ', number_of_correct_guesses)
    print('number of incorrect guesses: ', number_of_incorrect_guesses)
    total = number_of_correct_guesses + number_of_incorrect_guesses
    accuracy_percent = number_of_correct_guesses / total * 100
    print('accuracy: ', accuracy_percent, '%')





# get feature and label data
train_set_file_path = 'data/test_set.h5' # windows users might need anaconda / conda for h5
train_label_file_path = 'data/train_label.txt.gz'

# x - features, y - labels
x = pd.read_hdf(train_set_file_path).to_numpy()
y = pd.read_csv(train_label_file_path, header=None, sep="\t")

# to be rigorous, let's do a 50 / 50 split, 50% training data, 50% unknown data
x_training = x[:500] # max 1000 available samples
y_training = convert_labels_to_numeric(y[1][:500])


"""
including the trained data

x_verification = x[:1000]
y_verification = convert_labels_to_numeric(y[1][:1000])
sample_size = 1000
"""

# not including the trained data
sample_size = 500
x_verification = x[500:1000]
y_verification = convert_labels_to_numeric(y[1][500:1000])


# set up random forest model

clf = RandomForestClassifier()

clf.fit(x_training, y_training)


# how did we do?
run_stats()


# RESULTS
# after 50 / 50 split
# ~72% accuracy including the trained data
# ~44.4% accuracy not including the trained data (2% variance on each run)

"""
Example of labeled data to verify stats:

[0, 1, 0, 2, 0, 0, 0, 3, 1, 0, 0, 3, 0, 0, 1, 2, 0, 1, 4, 0, 0, 1, 0, 0, 3, 2, 4, 0, 1, 3, 0, 0, 2, 0, 0, 1, 4, 0, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 3, 3, 1, 1, 3, 0, 0, 0, 4, 1, 2, 1, 4, 3, 2, 1, 0, 1, 1, 1, 2, 0, 1, 0, 1, 1, 0, 3, 3, 2, 2, 0, 0, 1, 0, 1, 1, 0, 3, 0, 1, 0, 1, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 4, 0, 2, 1, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 2, 4, 4, 2, 1, 0, 0, 1, 1, 1, 1, 0, 4, 0, 0, 0, 0, 4, 0, 0, 2, 0, 3, 3, 1, 4, 3, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 3, 0, 0, 3, 0, 0, 1, 3, 1, 1, 1, 4, 1, 1, 2, 1, 0, 0, 0, 1, 0, 0, 1, 4, 0, 4, 0, 3, 3, 4, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 1, 1, 1, 4, 0, 1, 3, 0, 1, 0, 1, 1, 1, 0, 4, 0, 1, 3, 2, 1, 0, 0, 2, 0, 2, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 4, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 3, 1, 0, 0, 0, 3, 0, 0, 3, 4, 0, 1, 3, 0, 0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 2, 0, 0, 1, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2, 0, 0, 4, 0, 0, 0, 3, 0, 1, 3, 0, 1, 1, 1, 3, 1, 3, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 4, 3, 0, 1, 4, 3, 2, 0, 1, 1, 3, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 4, 2, 0, 0, 1, 0, 3, 1, 4, 0, 2, 2, 2, 3, 1, 0, 3, 1, 0, 0, 0, 2, 4, 4, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2, 3, 0, 0, 4, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0, 2, 0, 4, 0, 4, 0, 1, 2, 0, 3, 0, 1, 0, 0, 1, 0, 0, 1, 4, 1, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 4, 3, 0, 0, 0, 4, 1, 1, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 4, 3, 1, 2, 0, 0, 1, 1, 4, 0, 0, 3, 0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 1, 1, 4, 1, 4, 0, 0, 2, 1, 1, 1, 0, 1, 1, 4, 2, 1, 2, 1, 1, 0, 0, 1, 1, 2, 0, 1, 3, 0, 1, 0, 0, 4, 0, 4, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 0, 0, 1, 4, 1, 1, 3, 0, 1, 1, 4, 1, 0, 0, 0, 3, 0, 0, 1, 3, 0, 1, 0, 2, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 3, 0, 1, 0, 4, 1, 0, 0, 1, 4, 0, 0, 1, 2, 0, 0, 1, 4, 2, 0, 1, 1, 1, 2, 0, 0, 1, 1, 0, 1, 3, 0, 0, 0, 1, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 3, 0, 3, 1, 0, 1, 4, 0, 1, 0, 3, 0, 2, 1, 1, 0, 4, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 3, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 3, 0, 3, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 3, 1, 0, 1, 4, 4, 0, 0, 3, 1, 1, 1, 4, 3, 2, 1, 4, 3, 4, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 4, 0, 1, 4, 0, 4, 4, 1, 1, 4, 0, 0, 1, 1, 0, 1, 4, 0, 1, 3, 3, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 4, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 4, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 2, 3, 0, 1, 0, 1, 0, 1, 4, 0, 0, 0, 3, 0, 0, 1, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 4, 3, 2, 1, 0, 0, 1, 3, 4, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 3, 1, 1, 0, 0, 0, 0, 1, 0, 4, 1, 0, 2, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1, 1]


B cell
T cell
B cell
Granulocyte
B cell
B cell
B cell
NK cell
T cell
B cell
B cell
NK cell
B cell
B cell
T cell
Granulocyte
B cell
T cell
Monocyte
B cell
B cell
T cell
B cell
B cell
NK cell
Granulocyte
Monocyte
B cell
"""
