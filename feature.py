import itertools
import pandas as pd
import numpy as np
import gate_feature
from data_preprocess import single_generate_graph_adj_and_feature
from sim_processing import get_syn_sim, sim_thresholding
from linear_feature import get_low_feature
##提取线性特征和非线性特征。
# get sim matrix and association
seq_sim_matrix = pd.read_csv("data2/MNDR-lncRNA functional similarity matrix.csv", header=0, index_col=0).values
str_sim_matrix = pd.read_csv("data2/MNDR-disease semantic similarity matrix.csv", header=0, index_col=0).values
association = pd.read_csv("data2/MNDR-lncRNA-disease associations matrix.csv", header=0, index_col=0)
#linear feature
feature_MFl, feature_MFd = get_low_feature(16, 0.01, pow(10, -4), association.values)
l = pd.DataFrame(feature_MFl)
l.index = association.index.tolist()
l.to_csv('./data2/data2_feature_MFl.csv')
d = pd.DataFrame(feature_MFd)
d.index = association.columns.to_list()
d.to_csv('./data2/data2_feature_MFd.csv')
#non-linear feature（非线性特征是根据融合后的相似性网络和关联矩阵中进行提取的）
l_threshold = [0.6]  # generate subgraph
d_threshold = [0.7]
epochs=[300]
c_sim, d_sim = get_syn_sim(association.values, seq_sim_matrix, str_sim_matrix, mode=1)
print(c_sim)
for s in itertools.product(l_threshold,d_threshold,epochs):
    # GATE
    print(s[0],s[1])
    l_network = sim_thresholding(c_sim, s[0])
    d_network = sim_thresholding(d_sim, s[1])
    l_adj, l_features = single_generate_graph_adj_and_feature(l_network, association.values)
    d_adj, d_features = single_generate_graph_adj_and_feature(d_network, association.values.T)
    l_embeddings = gate_feature.get_gate_feature(l_adj, l_features,s[2], 1)
    d_embeddings = gate_feature.get_gate_feature(d_adj, d_features,s[2], 1)

l_emb = pd.DataFrame(l_embeddings)
l_emb.index = association.index.tolist()
l_emb.to_csv('./data2/data2_nonl_feature.csv')
d_emb = pd.DataFrame(d_embeddings)
d_emb.index = association.columns.to_list()
d_emb.to_csv('./data2/data2_nond_feature.csv')

#feature merger
rna_feature = np.hstack((feature_MFl,l_embeddings))
disease_feature = np.hstack((feature_MFd,d_embeddings))

lncrna = pd.DataFrame(rna_feature)
lncrna.index = association.index.tolist()
lncrna.to_csv('./data2/rna_feature.csv')
disease = pd.DataFrame(disease_feature)
disease.index = association.columns.to_list()
disease.to_csv('./data2/disease_feature.csv')

print("Finished")