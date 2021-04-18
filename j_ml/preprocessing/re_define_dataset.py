import pandas as pd
import numpy as np
import os
import re

# clean cluster id
def extract_id(in_str):
    return re.sub(r'.*_','', in_str)

# clean annotation
def clean_annotation(in_str):
    return re.sub(r'\(.*\)', '', in_str)

def clean_target_df(d_df):
    return_df = d_df.drop(columns=['Unnamed: 0', 'Cell.Barcode'])
    return_df['ClusterID'] = return_df['ClusterID'].apply(extract_id)
    return_df['Annotation'] = return_df['Annotation'].apply(clean_annotation)
    return return_df

key_val_dic = {'Bladder_1': 'Bladder_1',
    'BoneMarrowcKit_1': 'BoneMarrowcKit_1',
    'BoneMarrowcKit_2': 'BoneMarrowcKit_2',
    'BoneMarrowcKit_3': 'BoneMarrowcKit_3',
    'MesenchymalStemCellsPrimary_1': 'MesenchymalStemCellsPrimary_1',
    'BoneMarrow_1': 'BoneMarrow_1',
    'BoneMarrow_2': 'BoneMarrow_2',
    'BoneMarrow_3': 'BoneMarrow_3',
    'Brain_1': 'Brain_1',
    'Brain_2': 'Brain_2',
    'EmbryonicMesenchyme_1': 'EmbryonicMesenchyme_1',
    'EmbryonicStemCells_1': 'EmbryonicStemCells_1',
    'FetalFemaleGonad_1': 'FetalFemaleGonad_1',
    'FetalBrain_1': 'FetalBrain_1',
    'FetalIntestine_1': 'FetalIntestine_1',
    'FetalLiver_1': 'FetalLiver_1',
    'FetalLung_1': 'FetalLung_1',
    'FetalStomach_1': 'FetalStomach_1',
    'Kidney_2': 'Fetal_Kidney_2',
    'Liver_1': 'Liver_1',
    'Liver_2': 'Liver_2',
    'Lung_1': 'Lung_1',
    'Lung_2': 'Lung_2',
    'Lung_3': 'Lung_3',
    'Male(fetal)Gonad_1': 'Male(fetal)Gonad_1',
    'MesenchymalStemCells_1': 'MesenchymalStemCells_1',
    'Muscle_1': 'Muscle_1',
    'NeonatalBrain_1': 'NeonatalBrain_1',
    'NeonatalBrain_2': 'NeonatalBrain_2',
    'NeonatalCalvaria_1': 'NeonatalCalvaria_1',
    'NeonatalCalvaria_2': 'NeonatalCalvaria_2',
    'NeonatalHeart_1': 'NeonatalHeart_1',
    'NeonatalMuscle_1': 'NeonatalMuscle_1',
    'NeonatalMuscle_2': 'NeonatalMuscle_2',
    'NeonatalRib_1': 'NeonatalRib_1',
    'NeonatalRib_2': 'NeonatalRib_2',
    'NeonatalRib_3': 'NeonatalRib_3',
    'NeonatalSkin_1': 'NeonatalSkin_1',
    'Ovary_1': 'Ovary_1',
    'Ovary_2': 'Ovary_2',
    'Pancreas_1': 'Pancreas_1',
    'PeripheralBlood_1': 'PeripheralBlood_1',
    'PeripheralBlood_2': 'PeripheralBlood_2',
    'PeripheralBlood_3': 'PeripheralBlood_3',
    'PeripheralBlood_4': 'PeripheralBlood_4',
    'PeripheralBlood_5': 'PeripheralBlood_5',
    'PeripheralBlood_6': 'PeripheralBlood_6',
    'Placenta_1': 'Placenta_1',
    'Placenta_2': 'Placenta_2',
    'Prostate_1': 'Prostate_1',
    'Prostate_2': 'Prostate_2',
    'SmallIntestine_1': 'SmallIntestine_1',
    'SmallIntestine_2': 'SmallIntestine_2',
    'SmallIntestine_3': 'SmallIntestine_3',
    'Spleen_1': 'Spleen_1',
    'Stomach_1': 'Stomach_1',
    'Testis_1': 'Testis_1',
    'Testis_2': 'Testis_2',
    'Thymus_1': 'Thymus_1',
    'TrophoblastStemCells_1': 'TrophoblastStemCells_1',
    'Uterus_1': 'Uterus_1',
    'Uterus_2': 'Uterus_2',
    'Kidney_1': 'Fetal_Kidney_1',
    'MammaryGland': 'MammaryGland.Virgin_4',
    'NeonatalPancreas_1': 'NeonatalPancreas_1'}

# main function
if __name__ == "__main__":
    target_df = pd.read_csv('MCA_CellAssignments.csv')
    target_df = clean_target_df(target_df)
    groups = target_df.groupby('Batch')
    keys = [key for key, _ in groups]
    outdir = './processed_dge/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for root, dirs, files in os.walk("rmbatch_dge", topdown=False):
        for name in files:
            file_dir = os.path.join(root, name)
            tmp_df = pd.read_csv(file_dir, delimiter=' ')
            group_key = tmp_df.columns[0].split('.')[0]
            if group_key in key_val_dic.keys():
                group_key = key_val_dic.get(group_key)
            if group_key in keys:
                target_df = groups.get_group(group_key)
                if group_key == 'NeonatalPancreas_1':
                    n_gro = target_df.groupby('Annotation')
                    i=1
                    anoo_to_id = {}
                    for group in n_gro:
                        anoo_to_id[group[0]]=i
                        i+=1
                    for key,value in anoo_to_id.items():
                        target_df.loc[(target_df['Annotation'] == key),'ClusterID']=value

                list_of_items = target_df['Cell.name'].to_list()
                no_target_name_list = []
                for col_name in tmp_df.columns:
                    if col_name not in list_of_items:
                        no_target_name_list.append(col_name)

                tmp_df.drop(columns=no_target_name_list, inplace=True)
                tmp_df = tmp_df.T
                for i in range(len(target_df)) :
                    row_data = target_df.iloc[i]
                    index_name = row_data[0]
                    target_id = row_data[1]
                    annotation = row_data[4]
                    tmp_df.loc[index_name, 'target_id'] = target_id
                    tmp_df.loc[index_name, 'annotation'] = annotation
                
                file_str = outdir+group_key+'.csv'
                tmp_df.to_csv(file_str)
            else:
                print('skip %s'%file_dir)
    
    print('done')