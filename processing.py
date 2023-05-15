import numpy as np
import torch
import scipy.io
import mne

fs = 250

DICT_1 = {1: '1023', 2: '1072', 3: '276', 4: '277', 5: '32766', 6: '768', 7: '769', 8: '770', 9: '771', 10: '772'}
DICT_2 = {'1023': 'Rejected Trial', '1072': 'Eye Movements', '276': 'Idling EEG (eyes open)', '277': 'Idling EEG (eyes closed)',
'32766': 'Start of new run', '768': 'Start of a trial', '769': 'Cue Onset Left', '770': 'Cue Onset Right', 
'771': 'Cue Onset Foot', '772': 'Cue Onset Tongue'}

# (1, 1, N_ch, T)

def get_data(subject_id = 1):
	filename = 'A0' + str(subject_id) + 'T.gdf'
	raw=mne.io.read_raw_gdf(filename)

	arr = torch.from_numpy(raw.get_data())
	arr = torch.unsqueeze(arr, dim=0)
	arr = torch.unsqueeze(arr, dim=0)

	return arr

arr = get_data(1)

print('Data Shape: ', arr.shape)
#print(arr[0,0,0])

def get_labels(subject_id = 1):
	filename = 'A0' + str(subject_id) + 'T.gdf'
	raw=mne.io.read_raw_gdf(filename)

	time_points = mne.events_from_annotations(raw)[0][:, 0][7:]
	events = mne.events_from_annotations(raw)[0][:, 2][7:]

	return time_points, events

time_points, events = get_labels(1)


def get_nth_trial_data(n, time_points, events, data):

	start_idx = [i for i, n in enumerate(events) if n == 6][n] + 1
	end_idx = [i for i, n in enumerate(events) if n == 6][n+1]
	start_time = time_points[start_idx]
	end_time = time_points[end_idx]

	return data[:, :, :, start_time:end_time]

def get_nth_trial_label(n, events):

	start_idx = [i for i, n in enumerate(events) if n == 6][n] + 1

	return DICT_1[events[start_idx]]

print(get_nth_trial_data(0, time_points, events, arr).shape)
print(get_nth_trial_label(0, events))


