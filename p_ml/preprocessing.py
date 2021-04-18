import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()





class Preprocessing:


    """
    def __init__(self, train_set_file_path, train_label_file_path, test_set_file_path):
        self.train_set_file_path = train_set_file_path
        self.train_label_file_path = train_label_file_path
        self.test_set_file_path = test_set_file_path
    """


    def load_data(self):
        self.train_set = pd.read_hdf(self.train_set_file_path)
        self.train_label = pd.read_csv(self.train_label_file_path, header=None, sep="\t")
        self.test_set = pd.read_hdf(self.test_set_file_path, key="dge")


    def scale_sets(self, sets):
        # Get common genes, normalize  and scale the sets
        # input -- a list of all the sets to be scaled
        # output -- scaled sets
        common_genes = set(sets[0].index)
        for i in range(1, len(sets)):
            common_genes = set.intersection(set(sets[i].index),common_genes)
        common_genes = sorted(list(common_genes))
        sep_point = [0]
        for i in range(len(sets)):
            sets[i] = sets[i].loc[common_genes,]
            sep_point.append(sets[i].shape[1])
        total_set = np.array(pd.concat(sets, axis=1, sort=False), dtype=np.float32)
        total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
        total_set = np.log2(total_set+1)
        expr = np.sum(total_set, axis=1)
        total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
        cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
        total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
        for i in range(len(sets)):
            sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
        return sets


    def type_to_label_dict(self, types):
        # Make types to labels dictionary
        # input -- types
        # output -- type_to_label dictionary
        type_to_label_dict = {}
        all_type = list(set(types))
        for i in range(len(all_type)):
            type_to_label_dict[all_type[i]] = i
        return type_to_label_dict

    
    def one_hot_matrix(self, labels, C):
        # Turn labels into matrix
        # input -- labels (true labels of the sets), C (# types)
        # output -- one hot matrix with shape (# types, # samples)
        C = tf.constant(C, name = "C")
        one_hot_matrix = tf.one_hot(labels, C, axis = 0)
        sess = tf.Session()
        one_hot = sess.run(one_hot_matrix)
        sess.close()
        return one_hot

    
    def convert_type_to_label(self, types, type_to_label_dict):
        # Convert types to labels
        # input -- list of types, and type_to_label dictionary
        # output -- list of labels
        types = list(types)
        labels = list()
        for type in types:
            labels.append(type_to_label_dict[type])
        return labels


    def process_data(self):
        # set indexes
        self.train_set.index = [s.upper() for s in self.train_set.index]
        self.test_set.index = [s.upper() for s in self.test_set.index]

        # take care of duplicates
        self.train_set = self.train_set.loc[~self.train_set.index.duplicated(keep='first')]
        self.test_set = self.test_set.loc[~self.test_set.index.duplicated(keep='first')]

        barcode = list(self.test_set.columns)
        nt = len(set(self.train_label.iloc[:,1]))

        # scale the data        
        self.train_set, self.test_set = self.scale_sets([self.train_set, self.test_set])

        # create labels
        type_to_label_dict = self.type_to_label_dict(self.train_label.iloc[:,1])
        label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}

        print("Cell Types in training set:", type_to_label_dict)
        print("# Training cells:", self.train_label.shape[0])

        self.train_label = self.convert_type_to_label(self.train_label.iloc[:,1], type_to_label_dict)
        self.train_label = self.one_hot_matrix(self.train_label, nt)


    def run(self):
        self.train_set_file_path = 'data/train_set.h5'
        self.train_label_file_path = 'data/train_label.txt.gz'
        self.test_set_file_path = 'data/test_set.h5'

        self.load_data()
        self.process_data()


    def get_train_set(self):
        return self.train_set


    def get_train_label(self):
        return self.train_label


    def get_test_set(self):
        return self.test_set



