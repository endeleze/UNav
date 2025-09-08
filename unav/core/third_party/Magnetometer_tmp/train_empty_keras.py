from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D
import sys
import numpy as np

def get_model_1(n_inputs, n_outputs, loss_func='mae'):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(5, input_dim=20, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss=loss_func, optimizer='adam')
	return model

sys.path.append("../Magnetometer")
sys.path.append('../Magnetometer/processing_blocks_master/')
sys.path.append('../Magnetometer/processing_blocks_master/spectral_analysis/')

from preprocess import get_spectral_analysis_v5_features

batch_sample = np.random.sample(150).reshape(50,3)
features = get_spectral_analysis_v5_features( batch_sample )
num_inputs = len(features)
print(num_inputs)
get_model_1( num_inputs , 2, loss_func='mse').save('unfitted_model.keras')
