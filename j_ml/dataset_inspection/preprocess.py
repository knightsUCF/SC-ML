import pandas as pd
import numpy as np
import os

new_pre_process_dir = '/home/jay/Documents/projects/todo_files/ml_final_pro/datasets/pre_processed_datasets'

if not os.path.exists(new_pre_process_dir):
    os.mkdir(new_pre_process_dir)


# to prevent memory leak in order to process large dataset
# preprocess the dataset to binary for sequence in gene
for root, dirs, files in os.walk("/home/jay/Documents/projects/todo_files/ml_final_pro/datasets/processed_dge", topdown=False):
     for name in files:
        file_dir = os.path.join(root, name)
        tmp_df = pd.read_csv(file_dir)
        num_of_rows = tmp_df.shape[0]
        step_len = int(num_of_rows/2)
        for i in range(2):
            if i==0:
                tmp_df.iloc[:step_len,1:-2] = tmp_df.iloc[:step_len,1:-2].mask(tmp_df.iloc[:step_len,1:-2] > 1, 1)
            else:
                tmp_df.iloc[step_len:,1:-2] = tmp_df.iloc[step_len:,1:-2].mask(tmp_df.iloc[step_len:,1:-2] > 1, 1)
        new_file_dir = os.path.join(new_pre_process_dir, name)
        tmp_df.to_csv(new_file_dir, index=False)
        print('completed %s'%new_file_dir)

print('done')