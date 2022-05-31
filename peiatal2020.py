# -*- coding: utf-8 -*-
import sys
import subprocess
import os
import numpy as np
import pandas as pd
np.random.seed(56)  # for reproducibility
 
from keras.models import Model 
from keras.layers import Dense, Input, Dropout
from keras import regularizers

#============================================
# The supplementary material sets the batch nsize to 32, L1 regularization 10^-12, latent vector (encoding dim) to 200 the learning rate to 0.001 and noisy factor to 0.01
latent_dim = 200

batch_size_n = 32

learning_rate = 0.001

noise_factor = 0.01
#============================================

#output_file = "matrix_top10k_markers"

#os.mkdir("AE_"+output_file)

#mf_file = os.path.join('', output_file)
#mf_df = pd.read_table(mf_file, index_col=0)

mf_df = pd.read_csv(r'Q:\AUH-HAEM-FORSK-MutSigDLBCL222\external_data\PCAWG\mut_matrices\WGS_PCAWG.1536.csv')
penta = mf_df['Pentanucleotide']
mutation = mf_df['Mutation type']
mf_df = mf_df.drop(['Pentanucleotide', 'Mutation type'], axis = 1)

#data preprocessing
max_val = mf_df.max().max()

print("Data shape")
print(mf_df.shape)


np.random.seed(56)

test_set_percent = 0.2

# I deres README på github linker de til det ICGC repos hvor de henter PCAWG datamatricen. Den er organiseret således at patienterne er kolonnerne og mutationstyperne er rækkerne
# Derudover har de et preprocessing step der består af at dividere alle indgange med max-værdien (for at normalisere mellem 0 og 1) og navngivning af kolonnerne.
# https://github.com/bsml320/DeepMS/blob/master/Preprocess/SBS_preprocess.R


x_test = mf_df.sample(frac=test_set_percent)/max_val
#x_test = mf_df
x_train = mf_df.drop(x_test.index)/max_val
#x_train = mf_df
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size = x_train.shape)

x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size = x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)

x_test_noisy = np.clip(x_test_noisy, 0., 1.)

print("Dimensions of training and test data")
print(x_train_noisy.shape)
print(x_test_noisy.shape)


original_dim = mf_df.shape[1]

epochs_n = 50

# Compress to 200 dim
encoding_dim = latent_dim
 
# this is our input placeholder
input_dim = Input(shape=(original_dim,))
 
# encode
encoder_output = Dense(encoding_dim, activation = "relu", activity_regularizer = regularizers.l1(1e-12))(input_dim)

# decode
decoded = Dense(original_dim, activation = "softmax")(encoder_output)
 
# autoencoder model
autoencoder = Model(inputs = input_dim, outputs = decoded)
 
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
 
# training
hist = autoencoder.fit(x_train_noisy, x_train, epochs=epochs_n, batch_size=batch_size_n, shuffle=True, validation_data=(x_test_noisy, x_test), verbose = 0)
history_df = pd.DataFrame(hist.history)


# encoder model
encoder = Model(inputs = input_dim, outputs = encoder_output)

encoded_df = encoder.predict_on_batch(mf_df)
encoded_df = pd.DataFrame(encoded_df, index = mf_df.index)

encoded_df.index.name = 'sample_id'
encoded_df.columns.name = 'sample_id'

encoded_df.columns = encoded_df.columns + 1

# create a placeholder for an encoded (32-dimensional) input

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model( encoded_input, decoder_layer(encoded_input))

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
weights = []
for layer in encoder.layers:
        weights.append(layer.get_weights())

#weight_layer_df = pd.DataFrame(weights[1][0], columns=mf_df.columns, index=range(1, latent_dim+1))
weight_layer_df = pd.DataFrame(np.transpose(weights[1][0]), columns=mf_df.columns, index=range(1, latent_dim+1))
print("Weight shape and head")
print(weight_layer_df.shape)
print(weight_layer_df.head())
#========================
weights = []
for layer in decoder.layers:
        weights.append(layer.get_weights())

#weight_layer_df = pd.DataFrame(weights[1][0], columns=mf_df.columns, index=range(1, latent_dim+1))
weight_layer_df = pd.DataFrame(weights[1][0], columns=mf_df.columns, index=range(1, latent_dim+1))
print("Weight shape and head 2")
print(weight_layer_df.shape)
print(weight_layer_df.head())

