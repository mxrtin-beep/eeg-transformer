import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy as np

import processing as pro
from model import ContinuousTransformer


n_classes = 4
model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=16, dim_feedforward=32, dropout=0.2, n_classes=n_classes)

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


#excluded = []
def test(model, model_number, epoch, subject, normalize, bandpass_filter, excluded_y):

	print(f'Testing Model {model_number}, Epoch {epoch} on Subject {subject}')
	X, Y = pro.get_subject_dataset(subject, normalize=normalize, bandpass_filter=bandpass_filter, excluded=excluded_y)
	

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


	decoded_preds = decode_y(preds)

	#print(Y[:10])
	#print(preds[:10])

	fig, ax = plt.subplots()
	cm = confusion_matrix(Y, decoded_preds)
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
	cm_display.plot()
	
	plt.title('Confusion Matrix')
	plt.ylabel('Targets')
	plt.xlabel('Predictions')
	plt.savefig('charts/charts_' + str(model_number) + '/' + str(model_number) + '_' + str(epoch) + '_confusionmatrix.png')	
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


for i in range(1000, 7000+1000, 1000):
	test(model, model_number=3, epoch=i, subject=9, normalize=True, bandpass_filter=False, excluded_y=[])







