import numpy as np
import pandas as pd



f_r = pd.read_csv('./data2/rna_feature.csv',header=0,index_col=0).values
f_d = pd.read_csv('./data2/disease_feature.csv',header=0,index_col=0).values

all_associations = pd.read_csv('./data2' + '/pair.txt', sep=' ', names=['r', 'd', 'label'])

label = pd.read_excel('./data2/MNDR-lncRNA-disease associations matrix.xls',header=0,index_col=0)

label.to_csv("./data2/label.csv",header=None,index=None)





dataset = []

for i in range(int(all_associations.shape[0])):
    r = all_associations.iloc[i, 0]
    c = all_associations.iloc[i, 1]
    label = all_associations.iloc[i, 2]
    dataset.append(np.hstack((f_r[r], f_d[c], label)))

all_dataset = pd.DataFrame(dataset)

all_dataset.to_csv("./data2/data.csv",header=None,index=None)

print("Fnished!")