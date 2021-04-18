import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import HDF5Matrix
import math
import os


def create_superct_model(n_features):
    model = Sequential()
    model.add(Dense(200, input_dim = n_features, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'relu'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_dataset_from_dir(file_dir):
    d_df = pd.read_csv(file_dir)
    X = d_df.iloc[:,1:-2]
    y = d_df['target_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def dictionary_list (input_dic):
    return_str = ''
    for k, v in input_dic.items():
        return_str+=str(k)
        return_str+='\n\t'
        float_list = map(str, v) 
        return_str+='\n\t'.join(float_list)
        return_str+='\n'
        return_str+='\n'
    return return_str

def train_and_save(file_dir):
    print('start to process %s'%file_dir)
    X_train, X_test, y_train, y_test = load_dataset_from_dir(file_dir)
    batch_size = 128
    # round up the epches
    num_epoches = math.ceil(X_train.shape[0]/batch_size)
    model = create_superct_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=num_epoches, validation_data=(X_test, y_test))
    file_name = os.path.basename(file_dir)
    file_name = 'superct_v0_'+file_name
    history_name = file_name.replace('.csv', '.txt')
    output_model_name = file_name.replace('.csv', '.hdf5')
    with open(history_name, 'w') as f:
        f.write(file_name)
        f.write('\n')
        f.write('the number of observations: %d'%X_train.shape[0])
        f.write('\n')
        f.write('the number of features: %d'%X_train.shape[1])
        f.write('\n')
        f.write(dictionary_list(history.history))
    model.save(output_model_name)
    print('finished processing %s'%file_name)
    



if __name__ == "__main__":
    # processed_dge folder contains all new defined datasets
    # get all shapes from processed_dge files
    for root, dirs, files in os.walk("/home/jay/Documents/projects/todo_files/ml_final_pro/datasets/pre_processed_datasets", topdown=False):
        for name in files:
            file_dir = os.path.join(root, name)
            train_and_save(file_dir)