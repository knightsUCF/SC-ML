import data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


data = data.Data()



class Forest:


    def build(self):
        self.model = RandomForestClassifier()


    def train(self, x, y):
        x_train, self.x_test, y_train, self.y_test = train_test_split(x, y, test_size = 0.5)
        self.model.fit(x_train, y_train)


    def predict(self):
        pass


    def evaluate(self):
        print('Random Forest accuracy: ', self.model.score(self.x_test, self.y_test))
        feature_importances = self.model.feature_importances_
        feature_names = data.get_feature_names()
        self.important_features_df = pd.DataFrame(feature_names, index=feature_importances, columns = ['Gene names']).sort_index(ascending=False)
        print('important gene features: \n')
        print(self.important_features_df.head(10))
        # metrics for confusion matrix
        y_pred = self.model.predict(self.x_test)
        print("accuracy: ", metrics.accuracy_score(self.y_test, y_pred))
        self.y_pred = y_pred
        confusion_matrix = pd.crosstab(self.y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        sn.heatmap(confusion_matrix, annot=True)
        target_names_count = 1000


    def get_significant_features(self):
        return self.important_features_df


    def get_confusion_matrix(self):
        return (self.y_test, self.y_pred)
