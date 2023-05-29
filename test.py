import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn import metrics
import numpy as np

import processing as pro
from model import ContinuousTransformer




# DATA PARAMETERS
normalize = True
bandpass_filter = False
augment = False
load_data = False # False: save data
subjects = [1, 2, 3, 5, 6, 7, 8]
excluded_y = []


# TEST PARAMETERS
model_number = 5
epoch = 4000
subject = 9


n_classes = 4
model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=8, dim_feedforward=32, dropout=0.2, n_classes=n_classes)

def load_model(model_number, model, epoch):
	filename = 'models/models_' + str(model_number) + '/model_' + str(model_number) + '_epoch_' + str(epoch)+ '.pth'
	model.load_state_dict(torch.load(filename))
	#optimizer.load_state_dict(checkpoint["optimizer"])

	return model

def decode_y(Y):
	y_preds = []
	
	for i in range(len(Y)):
		pred_idx = torch.argmax(Y[i], dim=0)
		y_preds.append(pred_idx)

	return y_preds


def test(model, model_number, epoch, subject, normalize, bandpass_filter, excluded_y, augment):

	print(f'Testing Model {model_number}, Epoch {epoch} on Subject {subject}')

	X, Y = pro.get_testing_data(subject=subject, bandpass_filter=bandpass_filter, normalize=normalize, excluded_y=excluded_y, augment=augment,load_data=load_data)

	model = load_model(model_number, model, epoch)

	model.eval()
	with torch.no_grad():
		preds = model(X)
		#print(preds)
		#print(Y)
		loss = F.cross_entropy(preds, Y)

	#pro.print_predictions(preds, Y)

	percent = pro.percent_correct(preds, Y)

	print(f'Test Loss: {loss}')
	print(f'Percent Correct: {percent}')
	
	if len(Y.shape) > 1:
		preds = decode_y(preds)
		Y = decode_y(Y)

	precision = precision_score(Y, preds, average='macro')
	recall = recall_score(Y, preds, average='macro')
	print(f'Precision: {round(precision, 2)}')
	print(f'Recall: {round(recall, 2)}')

	fig, ax = plt.subplots()
	cm = confusion_matrix(Y, preds)
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
	cm_display.plot()
	
	plt.title('Confusion Matrix')
	plt.ylabel('Targets')
	plt.xlabel('Predictions')
	plt.savefig('charts/charts_' + str(model_number) + '/' + str(model_number) + '_' + str(epoch) + '_confusionmatrix.png')	
	plt.close()
	return loss, percent


def get_max_accuracy(model, model_number, subject):

	min_loss = 1000
	max_percent = 0

	for i in range(1000, 11000, 1000):
		loss, percent = test(model, model_number=model_number, epoch=i, subject=subject, normalize=True, bandpass_filter=False, excluded_y=[])
		if loss < min_loss:
			min_loss = loss
		if percent > max_percent:
			max_percent = percent

	return min_loss, max_percent


test(model, model_number=model_number, epoch=epoch, subject=subject, normalize=normalize, bandpass_filter=bandpass_filter, excluded_y=[], augment=augment)






