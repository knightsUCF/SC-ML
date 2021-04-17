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

random_forest_count = 0

def run_stats(clf):
    global random_forest_count
    global sample_size
    print('running stats for random forest: ', random_forest_count)
    number_of_correct_guesses = 0
    number_of_incorrect_guesses = 0
    i = 0

    for data_point in range(sample_size):
        sample = np.asarray(x_verification[i])
        predicted_sample = clf.predict([sample])
        # print(predicted_sample[0])
        # print(y_verification[i])
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

    random_forest_count += 1





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

sample_size = 1000
x_verification = x[:1000]
y_verification = convert_labels_to_numeric(y[1][:1000])
"""

# not including the trained data
sample_size = 500
x_verification = x[500:1000]
y_verification = convert_labels_to_numeric(y[1][500:1000])


# set up random forest model

# clf = RandomForestClassifier()


# tuning the model

# parameters we can fine tune:
# (unoptimized parameter return 44.2% accuracy)







def run_parameters_on_random_forest(_bootstrap, _ccp_alpha,
                              _class_weight,
                              _criterion,
                              _max_depth,
                              _max_features,
                              _max_leaf_nodes,
                              _max_samples,
                              _min_impurity_decrease,
                              _min_impurity_split,
                              _min_samples_leaf,
                              _min_samples_split,
                              _min_weight_fraction_leaf,
                              _n_estimators,
                              _n_jobs,
                              _oob_score,
                              _random_state,
                              _verbose,
                              _warm_start):

    clf = RandomForestClassifier(bootstrap=_bootstrap, ccp_alpha=_ccp_alpha, class_weight=_class_weight,
                                 criterion=_criterion, max_depth=_max_depth, max_features=_max_features,
                                 max_leaf_nodes=_max_leaf_nodes, max_samples=_max_samples,
                                 min_impurity_decrease=_min_impurity_decrease, min_impurity_split=_min_impurity_split,
                                 min_samples_leaf=_min_samples_leaf, min_samples_split=_min_samples_split,
                                 min_weight_fraction_leaf=_min_weight_fraction_leaf, n_estimators=_n_estimators,
                                 n_jobs=_n_jobs, oob_score=_oob_score, random_state=_random_state,
                                 verbose=_verbose, warm_start=_warm_start)

    clf.fit(x_training, y_training)

    # how did we do?
    run_stats(clf)



def search_for_optimal_parameters():
    results = {}
    stats = []

    max_depth_step = 5
    max_depth_upper_bound = 50

    max_leaf_nodes_step = 2


    """
    Random Forest Hyperparameters we’ll be Looking at:
    max_depth
    min_sample_split
    max_leaf_nodes
    min_samples_leaf
    n_estimators
    max_sample (bootstrap sample)
    max_features

    https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
    """

    _bootstrap = True
    _ccp_alpha = 0.0
    _class_weight = None
    _criterion = 'gini'
    _max_features = 'auto'
    _max_leaf_nodes = None
    _max_samples = None
    _min_impurity_decrease = 0.0
    _min_impurity_split = None
    _min_samples_leaf = 1  # 1 + i # 1 standard
    _min_samples_split = 2
    _min_weight_fraction_leaf = 0.0
    _n_estimators = 100
    _n_jobs = None
    _oob_score = False
    _random_state = None
    _verbose = 0
    _warm_start = False

    print('running new model for parameters: ')

    # max depth
    for i in range(50):
        _max_depth = i * max_depth_step # standard: None
        if _max_depth == 0:
            _max_depth = 1
        # cap upper bound
        if (_max_depth >= max_depth_upper_bound):
            _max_depth = max_depth_upper_bound

        print('max depth: ', _max_depth)

        # max leaf nodes
        for j in range(10):
            # max leaf nodes: standard: None
            _max_leaf_nodes = j * max_leaf_nodes_step  # standard: None
            if _max_leaf_nodes == 0 or _max_leaf_nodes == 1:
                _max_leaf_nodes = 2 # max leaf nodes must be greater than 1
            # cap upper bound
            max_leaf_nodes_upper_bound = 50
            if (_max_leaf_nodes >= max_leaf_nodes_upper_bound):
                _max_leaf_nodes = max_leaf_nodes_upper_bound
            print('max leaf nodes: ', _max_leaf_nodes)



            run_parameters_on_random_forest(_bootstrap, _ccp_alpha,
                                          _class_weight,
                                          _criterion,
                                          _max_depth,
                                          _max_features,
                                          _max_leaf_nodes,
                                          _max_samples,
                                          _min_impurity_decrease,
                                          _min_impurity_split,
                                          _min_samples_leaf,
                                          _min_samples_split,
                                          _min_weight_fraction_leaf,
                                          _n_estimators,
                                          _n_jobs,
                                          _oob_score,
                                          _random_state,
                                          _verbose,
                                          _warm_start)




search_for_optimal_parameters()

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
