import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import signatureanalyzer as sa

path = 'K:/FORSK-Projekt/Projekter/Scientific Projects/223_PhD_Ida/generated_data/DLBCL1001_trainset1_80p.csv'

sa.run_maf(path, outdir='K:/FORSK-Projekt/Projekter/Scientific Projects/223_PhD_Ida/generated_data/SA_outputDLBCL1001_set1/', reference='cosmic2', hg_build='./ref/hg19.2bit', nruns=10)
