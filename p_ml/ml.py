import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout



class ML:


    def create_superct_model(self):
        self.batch_size = 25
        self.epochs = 40
        n_features = 13331
        self.model = Sequential()
        self.model.add(Dense(200, input_dim = n_features, activation = 'relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(100, activation = 'relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(5, activation = 'relu'))
        self.model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])


    def load_data(self, train_set, train_label, test_set):
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set


    def train_model(self):
        self.train_set = self.train_set.T
        self.train_label = self.train_label.T
        self.model.fit(self.train_set, self.train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1) #, validation_data=())



