import numpy as np
import h5py
import torch

filename = 'sub-088_task-rest_eeg.set'


# (1, 1, N_ch, T)
def load_data(filename):

	a = h5py.File(filename)

	data = np.array(a['data'])
	data = torch.from_numpy(data)
	T, N_ch = data.shape[0], data.shape[1]

	return data.reshape(1, 1, N_ch, T)


data = load_data(filename)
print(data.shape)
