from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization



class ML:

    def load(self):
        self.model = Sequential()

        # layer 1
        self.model.add(Dense(200, activation='relu', input_shape=(784,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        # layer 2
        self.model.Add(Dense(100, activation='relu'))
        self.model.Add(BatchNormalization())
        self.model.add(Dropout(0.2))
        
        self.model.compile(loss='cateogrical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
 
        
        
 '''

Goals:

- design neural network with same parameters as outline

- Keras API

- fully connected ANN model

- to enhance the compatibility across different scRNA-seq platforms,
we convert the digital expression values to the binary values,
which means the genes are either present or absent in the cells

- input layer is connected to a hidden dense layer with 200 neurons,
and the first layer is fully connected to the next 100 neurons,
using ReLU activation function

- two random neuron dropouts occur after each layer in order to control over fitting 

- to avoid under representation of small sample sized cell types in the calculation of the accuracy function they include
class weight based on the sample size of each type in the model training 

- the loss function is defined as categorical cross entropy

Initial training and continual learning of the models 

- batch learning 

- training data sets 

- epochs 

 Transfer Learning in the model expansions 

- freezing hidden layers, and assessing weights for characterization of more cell types 

'''
        
 
