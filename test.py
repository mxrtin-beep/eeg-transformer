import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import processing as pro
from model import ContinuousTransformer



model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=512, dropout=0.1)

def load_model(model_number, model, epoch):
	filename = 'models/models_' + str(model_number) + '/model_' + str(model_number) + '_epoch_' + str(epoch)+ '.pth'
	model.load_state_dict(torch.load(filename))
	#optimizer.load_state_dict(checkpoint["optimizer"])

	return model

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


	
		
	return loss, percent


test(model, model_number=9, epoch=5999, subject=9, normalize=True, bandpass_filter=False)








