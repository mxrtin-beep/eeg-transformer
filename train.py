import torch
from torchtext.data import Field, BucketIterator
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

import processing as pro
import documentation as doc
from model import ContinuousTransformer



# TO DO
#	Try inputting wavelet transform
#	Inline call of test function


# MODEL PARAMETERS
nhead = 8
dim_feedforward = 32
dropout = 0.2


# HYPERPARAMETERS
model_number = 7
learning_rate = 0.00002
num_epochs = 80000
batch_size = 64
optimizer_name = 'adam'


# DATA PARAMETERS
normalize = True
bandpass_filter = False
augment = False
load_data = False # False: save data
subjects = [1, 2, 3, 5, 6, 7, 8]
excluded_y = []


# DOCUMENTATION
document = True
print_training_data = True
validate_fraction = 100
save_charts = True




print('Building Transformer Model...')
model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, n_classes=4)



print('Training...')

def save_losses(epoch, epochs, model_number, val_losses, print_training_data, tr_losses):

	plt.figure()
	plt.plot(epochs, val_losses, label='val loss', color='blue')
	if print_training_data: plt.plot(epochs, tr_losses, label='train loss', color='red')
	plt.title('Validation Losses, Epoch ' + str(epoch))
	plt.xlabel('Epoch')
	plt.legend()
	plt.savefig('charts/charts_' + str(model_number) + '/losses_' + str(epoch) + '.png')
	plt.figure()
	plt.close('all')

def save_accuracies(epoch, epochs, model_number, val_accuracies, print_training_data, tr_accuracies):
	plt.figure()
	plt.plot(epochs, val_accuracies, label='val accuracy', color='blue')
	if print_training_data: plt.plot(epochs, tr_accuracies, label='train accuracy',color='red')
	plt.legend()
	plt.title('Validation Percent Correct, Epoch ' + str(epoch))
	plt.xlabel('Epoch')
	plt.savefig('charts/charts_' + str(model_number) + '/accuracies_' + str(epoch) + '.png')
	plt.figure()
	plt.close('all')



def train(model, model_number, num_epochs, batch_size, optimizer_name, learning_rate, 
		subjects, bandpass_filter, normalize, excluded_y, load_data,
		document=False, print_training_data=False, validate_fraction=100, save_charts=True):

	tr_losses = []
	tr_accuracies = []
	val_losses = []
	val_accuracies = []
	epochs = []

	newpath = 'models/models_' + str(model_number) + '/'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	newpath = 'charts/charts_' + str(model_number) + '/'
	if not os.path.exists(newpath):
		os.makedirs(newpath)


	print('Creating Notes File...')
	if document: doc.create_notes_file(model_number, model, learning_rate, normalize, bandpass_filter, num_epochs, batch_size, excluded_y)

	

	X_tr, Y_tr, X_val, Y_val, X_te, Y_te = pro.get_training_data(subjects, bandpass_filter, normalize, excluded_y, augment, load_data)


	if optimizer_name == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	if optimizer_name == 'asgd':
		optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
	
	
	# X_tr (218, 1, 22, 1125)
	# Y_tr (218, 4)

	for epoch in range(num_epochs):

		
		
		if epoch == 0 or (epoch + 1) % validate_fraction == 0:
			print(f"[Epoch {epoch + 1} / {num_epochs}]")

			model.eval()
			with torch.no_grad():

				if print_training_data:
					ix = torch.randint(0, X_tr.shape[0], (batch_size*5, ))
					Xb, Yb = X_tr[ix], Y_tr[ix]
					train_preds = model(Xb)

					train_loss = F.cross_entropy(train_preds, Yb)
					train_percent = pro.percent_correct(train_preds, Yb)

					print(f'Training Loss: {train_loss}')
					print(f'Training Accuracy: {train_percent}')

					tr_losses.append(train_loss)
					tr_accuracies.append(train_percent)


				val_preds = model(X_val)
				val_loss = F.cross_entropy(val_preds, Y_val)
				val_percent = pro.percent_correct(val_preds, Y_val)

				
				print(f'Validation Loss: {val_loss}')
				print(f'Validation Accuracy: {val_percent}')
				
				
				val_losses.append(val_loss)
				val_accuracies.append(val_percent)
				epochs.append(epoch+1)

			model.train()
		
		if save_charts and (epoch + 1) % 1000 == 0:
		
			torch.save(model.state_dict(), 'models/models_' + str(model_number) + '/model_' + str(model_number) + '_epoch_' + str(epoch+1) + '.pth')
			save_losses(epoch+1, epochs, model_number, val_losses, print_training_data,tr_losses)
			save_accuracies(epoch+1, epochs, model_number, val_accuracies, print_training_data, tr_accuracies)
			if document: doc.update_file(model_number, epochs[-1], tr_losses[-1], tr_accuracies[-1], val_losses[-1], val_accuracies[-1])
		
		ix = torch.randint(0, X_tr.shape[0], (batch_size, ))
		Xb, Yb = X_tr[ix], Y_tr[ix]
		#print(Xb.dtype)

		preds = model(Xb)
		loss = F.cross_entropy(preds, Yb)

		#print(f'Loss: {loss}')
		loss.backward()

		optimizer.step()
		#print(out.shape)

	return model


train(model, model_number, num_epochs, batch_size, optimizer_name, learning_rate, 
		subjects, bandpass_filter, normalize, excluded_y, load_data,
	 	document=document, print_training_data=print_training_data, validate_fraction=validate_fraction, save_charts=save_charts)




