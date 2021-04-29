import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd



class Net:


    def build(self):
        self.batch_size = 25
        self.epochs = 40

        n_features = 14063 # input shape

        self.model = Sequential()

        self.model.add(Dense(200, input_dim=n_features, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(5, activation='relu'))
        self.model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])


    def train(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

        self.history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test, y_test))
        

    def evaluate(self):
        # evaluation accuracy metrics happens at the end of train(), this is further for analysis
        """
        
        import matplotlib.pyplot as plt
        import seaborn as sn
        
        X_test = self.features_test
        y_test = self.targets_test
        y_pred = self.model.predict(X_test)
        print(y_test)
        print(y_pred)
        
        # confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        
        # confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # sn.heatmap(confusion_matrix, annot=True)
        
        print('/n/n')


        # plot training and validation accuracy values
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
        # plot training and validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        """
        pass
    
