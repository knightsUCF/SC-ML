import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import train_test_split




class Data:
    
    
    def __init__(self):
        self.features_df = pd.read_hdf('data/features.h5').T
        self.features = self.features_df.to_numpy()
        self.targets_df = pd.read_csv('data/targets.txt.gz', header=None,  sep="\t")
        self.targets = self.convert_to_numeric_labels(self.targets_df[1])
        print(self.features_df)
        print(self.targets_df)

        
    def convert_to_numeric_labels(self, data):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(data)
        return label_encoder.transform(data)


    def get_features(self):
        return self.features


    def get_targets(self):
        return self.targets


    def get_features_shape(self):
        return self.features.shape


    def get_targets_shape(self):
        return self.targets.shape


    def get_feature_names(self):
        return self.features_df.columns


    def get_target_names(self):
        return self.targets_df


    def describe(self):
        print('Features data: ', self.get_features())
        print('Feature names: ', self.get_feature_names())
        print('Targets data: ', self.get_targets())
        print('Target names: ', self.get_target_names())
        print('Features shape: ', self.get_features_shape())
        print('Targets shape: ', self.get_targets_shape())
