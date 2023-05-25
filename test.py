import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import numpy as np

import processing as pro
from model import ContinuousTransformer



model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=256, dropout=0.1)

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



def test(model, model_number, epoch, subject, normalize, bandpass_filter):

	print(f'Testing Model {model_number} on Subject {subject}')
	X, Y = pro.get_subject_dataset(subject, normalize=normalize, bandpass_filter=bandpass_filter)
	

	model = load_model(model_number, model, epoch)


	model.eval()
	with torch.no_grad():
		preds = model(X)
		#print(preds)
		#print(Y)
		loss = F.cross_entropy(preds, Y)

	pro.print_predictions(preds, Y)

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
	plt.savefig('charts/' + str(model_number) + '_confusionmatrix.png')	
	return loss, percent


test(model, model_number=9, epoch=8000, subject=9, normalize=True, bandpass_filter=False)








