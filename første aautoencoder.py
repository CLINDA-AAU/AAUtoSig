import keras
from keras import layers
import pandas as pd
import numpy as np

from keras.models import Model 
from keras.layers import Dense, Input, Dropout
from keras import regularizers



#Divide mutational count matrix into training and test sample

mc = pd.read_csv('Q:/AUH-HAEM-FORSK-MutSigDLBCL222/generated_data/WGS_PCAWG_96_LymphBNHL.csv', index_col=0).transpose()
mc_df = mc.copy()

n_patients, n_mut = mc_df.shape


x_train = mc_df.sample(frac = 0.8)
x_test = mc_df.drop(x_train.index)

x_train_array = x_train.to_numpy()
x_test_array = x_test.to_numpy()

n_patients_train = x_train.shape[0]


#hyperparameters
n_epochs = 50
latent_dim = 12 #n_sigs?
batch_size = 8



#One hidden layer autoencoer
input_dim = Input(shape=(n_mut,))

# "encoded" represents the encoidng of the input (dimension reduction)
encoded = Dense(latent_dim, activation='relu')(input_dim)

# "decoded" represents the reconstruction of the input
decoded = Dense(n_mut, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = Model(inputs = input_dim, outputs = decoded)


#loss function
autoencoder.compile(optimizer='adam', loss='mse')

#training
autoencoder.fit(x_train_array, x_train_array,
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test_array, x_test_array))



#Retrieve the enodced and decoded layer
encoder = Model(input_dim, encoded)
encoded_input = Input(shape=(latent_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


encoded_sigs = encoder.predict(x_test_array)
decoded_sigs = decoder.predict(encoded_sigs)