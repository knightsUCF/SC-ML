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


    def get_gene_counts_per_cell(self, cell_type):
        print('running counts per cell type: ', cell_type)
        df0 = self.get_features_df()
        df1 = self.get_targets_df()
        df1 = df1.set_index(df1.columns[0])
        df0 = df0.merge(df1, left_index=True, right_index=True)
        df0 = df0.rename(columns={1: 'targets'})

        cell_to_keep = [cell_type]

        df0 = df0[df0['targets'].isin(cell_to_keep)]

        Cd79a = df0['Cd79a'].sum()
        H2_Aa = df0['H2-Aa'].sum()
        H2_Eb1 = df0['H2-Eb1'].sum()
        Cd3d = df0['Cd3d'].sum()
        Cd3e = df0['Cd3e'].sum()
        Faim3 = df0['Faim3'].sum()
        Tyrobp = df0['Tyrobp'].sum()
        Fyb = df0['Fyb'].sum()
        Fcer1g = df0['Fcer1g'].sum()
        Tmem66 = df0['Tmem66'].sum()
        _2010001M09Rik = df0['2010001M09Rik'].sum()
        Mef2c = df0['Mef2c'].sum()
        Fxyd5 = df0['Fxyd5'].sum()
        Skap1 = df0['Skap1'].sum()
        Cd247 = df0['Cd247'].sum()
        Gzma = df0['Gzma'].sum()
        Nkg7 = df0['Nkg7'].sum()

        print('counts: ')
        print('Cd79a: ', Cd79a)
        print('H2-Aa: ', H2_Aa)
        print('H2-Eb1: ', H2_Eb1)
        print('Cd3d: ', Cd3d)
        print('Cd3e: ', Cd3e)
        print('Faim3: ', Faim3)
        print('Tyrobp: ', Tyrobp)
        print('Fyb: ', Fyb)
        print('Fcer1g: ', Fcer1g)
        print('Tmem66: ', Tmem66)
        print('2010001M09Rik: ', _2010001M09Rik)
        print('Mef2c: ', Mef2c)
        print('Fxyd5: ', Fxyd5)
        print('Skap1: ', Skap1)
        print('Cd247: ', Cd247)
        print('Gzma: ', Gzma)
        print('Nkg7: ', Nkg7)
        print('\n')

     
    def keep_only_b_and_t_cells(self):
        df0 = self.get_features_df()
        df1 = self.get_targets_df()
        df1 = df1.set_index(df1.columns[0])
        df0 = df0.merge(df1, left_index=True, right_index=True)
        df0 = df0.rename(columns={1: 'targets'})
        cells_to_keep = ['T cell' , 'B cell']
        df0 = df0[df0['targets'].isin(cells_to_keep)]
        df_targets = pd.DataFrame(df0[df0.columns[-1]], columns = ['targets'])
        self.targets = self.convert_to_numeric_labels(df_targets['targets'])
        df_features = df0.drop('targets', 1)
        self.features = df_features.to_numpy()


    def get_features_df(self):
        return self.features_df


    def get_targets_df(self):
        return self.targets_df

        
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


