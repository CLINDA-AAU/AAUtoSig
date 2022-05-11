# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

PCAWG = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\Alexandrov_2020_synapse\WGS_PCAWG_2018_02_09\WGS_PCAWG.96.csv')
PCAWG.index = [t[0] + '[' + m + ']' + t[2] for (t,m) in zip(PCAWG['Trinucleotide'], PCAWG['Mutation type'])]
PCAWG = PCAWG.drop(['Trinucleotide', 'Mutation type'], axis = 1)

cancers = [c.split('::')[0] for c in PCAWG.columns]
PatientID = [c.split('::')[1] for c in PCAWG.columns]

encoder = OneHotEncoder(sparse=(False))
cancer_one_hot_df_PCAWG = pd.DataFrame(encoder.fit_transform(np.array(cancers).reshape(-1, 1)))
cancer_one_hot_df_PCAWG.columns = encoder.get_feature_names()
cancer_one_hot_df_PCAWG.index = PatientID


TCGA = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\Alexandrov_2020_synapse\WES_TCGA_2018_03_09\WES_TCGA.96.csv')
TCGA.index = [t[0] + '[' + m + ']' + t[2] for (t,m) in zip(TCGA['Trinucleotide'], TCGA['Mutation type'])]
TCGA = TCGA.drop(['Trinucleotide', 'Mutation type'], axis = 1)

cancers = [c.split('::')[0] for c in TCGA.columns]
PatientID = [c.split('::')[1] for c in TCGA.columns]

cancer_one_hot_df_TCGA = pd.DataFrame(encoder.fit_transform(np.array(cancers).reshape(-1, 1)))
cancer_one_hot_df_TCGA.columns = encoder.get_feature_names()
cancer_one_hot_df_TCGA.index = PatientID
