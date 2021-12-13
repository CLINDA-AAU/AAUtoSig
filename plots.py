import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plotsigs(context, mutation, intensities):
    colors = {'C>A': 'r', 'C>G': 'b', 'C>T': 'g', 
              'T>A' : 'y', 'T>C': 'c','T>G' : 'm' }
    plt.figure(figsize=(20,7))
    plt.bar(x = context, 
            height =  intensities/np.sum(intensities), 
            color = [colors[i] for i in mutation])
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles,labels)
    plt.xticks(rotation=90)

