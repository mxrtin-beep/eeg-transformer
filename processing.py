import numpy as np
import torch
import scipy.io
import mne
import torch.nn.functional as F
import scipy.signal as sig
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt



fs = 250
low_freq = 4
high_freq = 40

T = 1125	# Number of time points that fit in the model
N_ch = 25	# Number of channels that fit in the model

'''

0: Left
1: Right
2: Foot
3: Tongue


'''

DICT_1 = {1: '1023', 2: '1072', 3: '276', 4: '277', 5: '32766', 6: '768', 7: '769', 8: '770', 9: '771', 10: '772'}
DICT_2 = {'1023': 'Rejected Trial', '1072': 'Eye Movements', '276': 'Idling EEG (eyes open)', '277': 'Idling EEG (eyes closed)',
'32766': 'Start of new run', '768': 'Start of a trial', '769': 'Cue Onset Left', '770': 'Cue Onset Right', 
'771': 'Cue Onset Foot', '772': 'Cue Onset Tongue'}

# (1, 1, N_ch, T)
mne.set_log_level(verbose='CRITICAL')



# --------------------------------------------------- HELPER METHODS -------------------------------------------------------------------------
def get_data(subject_id = 1):
	extension = 'BCICIV_2a_gdf'
	filename = 'A0' + str(subject_id) + 'T.gdf'
	raw=mne.io.read_raw_gdf(extension + '/' + filename)

	arr = torch.from_numpy(raw.get_data())
	arr = torch.unsqueeze(arr, dim=0)
	arr = torch.unsqueeze(arr, dim=0)

	return arr



def get_labels(subject_id = 1):
	extension = 'BCICIV_2a_gdf'
	filename = 'A0' + str(subject_id) + 'T.gdf'
	raw=mne.io.read_raw_gdf(extension + '/' + filename)

	time_points = mne.events_from_annotations(raw)[0][:, 0][7:]
	events = mne.events_from_annotations(raw)[0][:, 2][7:]

	return time_points, events


def get_nth_trial_data(n, time_points, events, data):

	start_idx = [i for i, n in enumerate(events) if n == 6][n] + 1
	end_idx = [i for i, n in enumerate(events) if n == 6][n+1]
	start_time = time_points[start_idx]
	end_time = time_points[end_idx]

	return data[:, :, :, start_time:end_time]

def get_nth_trial_label(n, events):

	start_idx = [i for i, n in enumerate(events) if n == 6][n] + 1

	return DICT_1[events[start_idx]]



def shuffle(X, Y):
	indices = torch.randperm(X.size()[0])
	return X[indices], Y[indices]



# --------------------------------------------------- DATASET METHODS -------------------------------------------------------------------------

n_augment = 50

def get_training_data(subject_list, bandpass_filter, normalize, excluded_y, augment, load_data):
	
	print(f'Applying Bandpass Filter: {bandpass_filter}.')
	print(f'Normalizing: {normalize}.')
	print(f'Augmenting: {augment}.')

	suffix = ''
	if normalize:
		suffix += '_norm'
	if augment:
		suffix += '_aug'

	if load_data:
		X_tr = torch.load('data/X_tr' + suffix + '.pt')
		Y_tr = torch.load('data/Y_tr' + suffix + '.pt')
		X_val = torch.load('data/X_val' + suffix + '.pt')
		Y_val = torch.load('data/Y_val' + suffix + '.pt')
		X_te = torch.load('data/X_te' + suffix + '.pt')
		Y_te = torch.load('data/Y_te' + suffix + '.pt')
		print('Loading Data: ', 'data/X_tr' + suffix + '.pt')

	# Saving data...
	else:

		X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_subject_wide_data(subject_list=subject_list, bandpass_filter=bandpass_filter, excluded=excluded_y, augment=augment)

		print(f'Excluding Y categories {excluded_y}...')
		X_tr, Y_tr = exclude_categories(X_tr, Y_tr, excluded_y)
		X_val, Y_val = exclude_categories(X_val, Y_val, excluded_y)
		X_te, Y_te = exclude_categories(X_te, Y_te, excluded_y)


		print('Input Shape: ', X_tr.shape)

		#X_tr, Y_tr = pro.shuffle(X_tr, Y_tr)
		#X_val, Y_val = pro.shuffle(X_val, Y_val)
		#X_te, Y_te = pro.shuffle(X_te, Y_te)

		if normalize:
			X_tr, X_val, X_te = normalize_data(X_tr, X_val, X_te)


		torch.save(X_tr, 'data/X_tr' + suffix + '.pt')
		torch.save(Y_tr, 'data/Y_tr' + suffix + '.pt')
		torch.save(X_val, 'data/X_val' + suffix + '.pt')
		torch.save(Y_val, 'data/Y_val' + suffix + '.pt')
		torch.save(X_te, 'data/X_te' + suffix + '.pt')
		torch.save(Y_te, 'data/Y_te' + suffix + '.pt')
		print('Saving Data ' + 'data/X_tr' + suffix + '.pt')

	return X_tr, Y_tr, X_val, Y_val, X_te, Y_te

def get_testing_data(subject, bandpass_filter, normalize, excluded_y, augment, load_data):

	print(f'Applying Bandpass Filter: {bandpass_filter}.')
	print(f'Normalizing: {normalize}.')
	print(f'Augmenting: {augment}.')

	suffix = ''
	if normalize:
		suffix += '_norm'
	if augment:
		suffix += '_aug'

	if load_data:
		X = torch.load('data/X_9' + suffix + '.pt')
		Y = torch.load('data/Y_9' + suffix + '.pt')
		print('Loading Data: ', 'data/X_9' + suffix + '.pt')
		print('Shape ', X.shape)

	else:
		X, Y, _, _, _, _ = get_subject_wide_data(subject_list=[subject], bandpass_filter=bandpass_filter, excluded=excluded_y, augment=augment)

		X, Y = exclude_categories(X, Y, excluded_y)

		if normalize:
			X = normalize_test_data(X)


		torch.save(X, 'data/X_9' + suffix + '.pt')
		torch.save(Y, 'data/Y_9' + suffix + '.pt')
		print('Saving Data: ', 'data/X_9' + suffix + '.pt')
		print('Shape: ', X.shape)


	return X, Y


def get_subject_dataset(subject_id = 1, bandpass_filter=False, excluded=[], augment=False):
	data = get_data(subject_id)
	time_points, events = get_labels(subject_id)

	
	X = torch.zeros((1, 1, N_ch, T), dtype=torch.float32)
	Y = torch.zeros((1), dtype=torch.long)

	n_trials = len([i for i, n in enumerate(events) if n == 6])	# Number of times the number 6 appears in events
	
	n = 0
	while (n+1) < n_trials:
		arr = get_nth_trial_data(n, time_points, events, data)

		if not augment:
			arr = arr[:, :, :N_ch, :T]		# Only take the first 22 channels and 1125 time points
		else:
			arr = arr[:, :, :N_ch, :]

		y_arr = np.array([int(get_nth_trial_label(n, events)) - 769])
		y = torch.tensor(y_arr)
		#print(y.shape)
		#print(arr.shape, n)
		
		while arr.shape[3] < T:	# If the array is too short, add time points
			diff = T - arr.shape[3]
			#print(diff)
			#print(arr.shape)
			arr = torch.cat((arr, arr[:, :, :, -diff:]), 3)
			#print('new shape: ', arr.shape)

		if y_arr[0] != 254:			# 1023: rejected trial, don't add.
			if not augment:
				X = torch.cat((X, arr), 0)
				Y = torch.cat((Y, y), 0)
			else:
				arr_length = arr.shape[3]
				diff = arr_length - T
				#print(f'To trial {n} adding {int(diff/n_augment)} points.')
				for i in range(0, diff, n_augment):
					aug_arr = arr[:, :, :, i:i+T]
					X = torch.cat((X, aug_arr), 0)
					Y = torch.cat((Y, y), 0)



		n += 1

	X = X[1:, :, :, :].to(torch.float32)
	Y = Y[1:]

	X, Y = exclude_categories(X, Y, excluded)
	print(f'Subject {subject_id}: {X.shape}.')

	return X, Y



def get_split_data(subject_id = 1, bandpass_filter=False, excluded=[], augment=False):

	X, Y = get_subject_dataset(subject_id, excluded=excluded, augment=augment)

	Y = F.one_hot(Y, num_classes=(4 - len(excluded)))

	#X, Y = shuffle(X, Y)

	n1 = int(0.8*X.shape[0])
	n2 = int(0.9*X.shape[0])

	X_tr, Y_tr = X[:n1, :, :, :], Y[:n1]
	X_val, Y_val = X[n1:n2, :, :, :], Y[n1:n2]
	X_te, Y_te = X[n2:, :, :, :], Y[n2:]

	if bandpass_filter:
		X_tr = bandpass_filter_data(X_tr, low_freq, high_freq, fs)
		X_val = bandpass_filter_data(X_val, low_freq, high_freq, fs)
		X_te = bandpass_filter_data(X_te, low_freq, high_freq, fs)


	return X_tr.to(torch.float32), Y_tr.to(torch.float32), X_val.to(torch.float32), Y_val.to(torch.float32), X_te.to(torch.float32), Y_te.to(torch.float32)



def get_subject_wide_data(subject_list, bandpass_filter=False, excluded=[], augment=False):

	X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_split_data(subject_list[0], bandpass_filter=bandpass_filter, excluded=excluded, augment=augment)
	arr = [X_tr, Y_tr, X_val, Y_val, X_te, Y_te]
	

	for i in range(1, len(subject_list)):
		X_tri, Y_tri, X_vali, Y_vali, X_tei, Y_tei = get_split_data(subject_list[i], bandpass_filter=bandpass_filter, excluded=excluded, augment=augment)
		brr = [X_tri, Y_tri, X_vali, Y_vali, X_tei, Y_tei]

		for j in range(len(arr)):
			arr[j] = torch.cat((arr[j], brr[j]), 0)
	
	print(f'Acquired {arr[0].shape[0]} data points from subjects {subject_list}.')
	return arr







# --------------------------------------------------- TEST DATA METHODS -------------------------------------------------------------------------


def percent_correct(preds, targets):

	assert(len(preds) == len(targets))
	count = 0
	for i in range(len(preds)):
		pred_idx = torch.argmax(preds[i], dim=0)

		#print(len(targets.shape))
		# If encoded or not
		if len(targets.shape) == 1:
			target_idx = targets[i]
		else:
			target_idx = torch.argmax(targets[i], dim=0)
		
		if pred_idx == target_idx:
			count += 1

	return round(count / len(preds) * 100, 4)


def print_predictions(preds, targets):
	correct = 0
	#print(preds, targets)
	for i in range(len(preds)):
		pred_idx = torch.argmax(preds[i], dim=0)
		if pred_idx != targets[i]:
			print(f'Prediction: {pred_idx}\t Target: {targets[i]} \t {correct}/{i+1}')
		else:
			correct += 1
			print(f'Prediction: {pred_idx}\t Target: {targets[i]} \t {correct}/{i+1}  \tCorrect!')







# --------------------------------------------------- MODIFY DATA METHODS -------------------------------------------------------------------------


def bandpass_filter_data(data, low, high, fs, order=5):
	nyquist = 0.5 * fs
	b, a = sig.butter(order, [low / nyquist, high / nyquist], btype='band')
	arr = sig.filtfilt(b, a, data)
	return torch.from_numpy(arr.copy())


def normalize_data(X_tr, X_val, X_te):

	N_ch = X_tr.shape[2]

	for j in range(N_ch):
		
		mu = torch.mean(X_tr[:, 0, j, :])
		sigma = torch.std(X_tr[:, 0, j, :])

		torch.save(mu, 'data/mu_' + str(j) + '.pt')
		torch.save(sigma, 'data/sigma_' + str(j) + '.pt')

		X_tr[:, 0, j, :] = (X_tr[:, 0, j, :] - mu) / sigma
		X_val[:, 0, j, :] = (X_val[:, 0, j, :] - mu) / sigma
		X_te[:, 0, j, :] = (X_te[:, 0, j, :] - mu) / sigma

	return X_tr, X_val, X_te

def normalize_test_data(X):

	N_ch = X.shape[2] # 25

	for j in range(N_ch): # 0 to 24

		mu_j = torch.load('data/mu_' + str(j) + '.pt')
		sigma_j = torch.load('data/sigma_' + str(j) + '.pt')

		X[:, 0, j, :] = (X[:, 0, j, :] - mu_j) / sigma_j

	return X


def exclude_categories(X, Y, excluded):

	x, y = X, Y
	for i in excluded:
		x, y = x[y != i], y[y != i]

	return x, y


# --------------------------------------------------- SPECTROGRAM -------------------------------------------------------------------------

#X_tr, _, X_val, _, X_te, _ = get_split_data(1)

#normalize_data(X_tr, X_val, X_te)


