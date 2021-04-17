import preprocessing
import ml

preprocessing = preprocessing.Preprocessing()
ml = ml.ML()



# run preprocessing on data
preprocessing.run() # here is when we can later pass in the different data file path

# send preprocessed data to the ml model
ml.load_data(preprocessing.get_train_set(), preprocessing.get_train_label(), preprocessing.get_test_set())

# create model
ml.create_superct_model()

# train model
ml.train_model()
