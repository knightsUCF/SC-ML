import h5py
import numpy as np
import pandas as pd

data_f = h5py.File('../../datasets/MCA_BatchRemoved_Merge_dge.h5ad', 'r')
# open a new file for storing processed dataset
processed_f = h5py.File('../../datasets/process_dataset.h5ad', 'w')
# copy dataset into the new dataset
print('start copying dataset')
data_f.copy('X', processed_f)
data_f.copy('var', processed_f)
data_f.copy('obs', processed_f)

print(' ')
print('copying completed')
# close the dataset in order to save memory cost
data_f.close()

print('start preprocessing')
i = 0
while i < processed_f['X'].shape[1]:
    tmp_array = processed_f['X'][:, i:i+3883]
    tmp_array = np.where(tmp_array > 1.0, 1.0, tmp_array)
    processed_f['X'][:, i:i+3883] = tmp_array
            
    i += 3883
    process_percent = (i+1)/processed_f['X'].shape[1]
    percentage = "{:.1%}".format(process_percent)
    print(percentage)

print('storing labels')

label_data = pd.read_csv('../../datasets/MCA_BatchRemoved_Merge_dge_cellinfo.csv')
label_data = label_data['louvain']
processed_f.create_dataset(name='labels', data = label_data, dtype=np.int)
processed_f.flush()

print('completed')

processed_f.close()
    
