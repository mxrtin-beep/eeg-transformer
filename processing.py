import numpy as np
import torch
import scipy.io
import mne
import torch.nn.functional as F

fs = 250

DICT_1 = {1: '1023', 2: '1072', 3: '276', 4: '277', 5: '32766', 6: '768', 7: '769', 8: '770', 9: '771', 10: '772'}
DICT_2 = {'1023': 'Rejected Trial', '1072': 'Eye Movements', '276': 'Idling EEG (eyes open)', '277': 'Idling EEG (eyes closed)',
'32766': 'Start of new run', '768': 'Start of a trial', '769': 'Cue Onset Left', '770': 'Cue Onset Right', 
'771': 'Cue Onset Foot', '772': 'Cue Onset Tongue'}

# (1, 1, N_ch, T)
mne.set_log_level(verbose='CRITICAL')

def get_data(subject_id = 1):
	extension = 'BCICIV_2a_gdf'
	filename = 'A0' + str(subject_id) + 'T.gdf'
	raw=mne.io.read_raw_gdf(extension + '/' + filename)

	arr = torch.from_numpy(raw.get_data())
	arr = torch.unsqueeze(arr, dim=0)
	arr = torch.unsqueeze(arr, dim=0)

	return arr

#arr = get_data(1)

#print('Data Shape: ', arr.shape)
#print(arr[0,0,0])

def get_labels(subject_id = 1):
	extension = 'BCICIV_2a_gdf'
	filename = 'A0' + str(subject_id) + 'T.gdf'
	raw=mne.io.read_raw_gdf(extension + '/' + filename)

	time_points = mne.events_from_annotations(raw)[0][:, 0][7:]
	events = mne.events_from_annotations(raw)[0][:, 2][7:]

	return time_points, events

#time_points, events = get_labels(1)


def get_nth_trial_data(n, time_points, events, data):

	start_idx = [i for i, n in enumerate(events) if n == 6][n] + 1
	end_idx = [i for i, n in enumerate(events) if n == 6][n+1]
	start_time = time_points[start_idx]
	end_time = time_points[end_idx]

	return data[:, :, :, start_time:end_time]

def get_nth_trial_label(n, events):

	start_idx = [i for i, n in enumerate(events) if n == 6][n] + 1

	return DICT_1[events[start_idx]]

#print(get_nth_trial_data(0, time_points, events, arr).shape)
#print(get_nth_trial_label(0, events))

def get_subject_dataset(subject_id = 1):
	data = get_data(subject_id)
	time_points, events = get_labels(subject_id)

	T = 1125	# Number of time points that fit in the model
	N_ch = 22	# Number of channels that fit in the model

	X = torch.zeros((1, 1, N_ch, T), dtype=torch.float32)
	Y = torch.zeros((1), dtype=torch.long)

	n_trials = len([i for i, n in enumerate(events) if n == 6])	# Number of times the number 6 appears in events
	
	n = 0
	while (n+1) < n_trials:
		arr = get_nth_trial_data(n, time_points, events, data)[:, :, :N_ch, :T]		# Only take the first 22 channels and 1125 time points
		y_arr = np.array([int(get_nth_trial_label(n, events)) - 769])
		y = torch.tensor(y_arr)
		#print(y.shape)
		#print(arr.shape, n)
		
		while arr.shape[3] != T:	# If the array is too short, add time points
			diff = T - arr.shape[3]
			#print(diff)
			#print(arr.shape)
			arr = torch.cat((arr, arr[:, :, :, -diff:]), 3)
			#print('new shape: ', arr.shape)

		if y_arr[0] != 254:			# 1023: rejected trial, don't add.
			X = torch.cat((X, arr), 0)
			Y = torch.cat((Y, y), 0)
		n += 1

	return X[1:, :, :, :].to(torch.float32), Y[1:]



def get_split_data(subject_id = 1):

	X, Y = get_subject_dataset(subject_id)

	Y = F.one_hot(Y, num_classes=4)

	n1 = int(0.8*X.shape[0])
	n2 = int(0.9*X.shape[0])

	X_tr, Y_tr = X[:n1, :, :, :], Y[:n1]
	X_val, Y_val = X[n1:n2, :, :, :], Y[n1:n2]
	X_te, Y_te = X[n2:, :, :, :], Y[n2:]

	return X_tr.to(torch.float32), Y_tr.to(torch.float32), X_val.to(torch.float32), Y_val.to(torch.float32), X_te.to(torch.float32), Y_te.to(torch.float32)




X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_split_data(1)

preds = torch.randn((32, 10))


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
			print(f'Prediction: {pred_idx}\t Target: {targets[i]} \t {correct}/{i}')
		else:
			correct += 1
			print(f'Prediction: {pred_idx}\t Target: {targets[i]} \t {correct}/{i}\tCorrect!')



