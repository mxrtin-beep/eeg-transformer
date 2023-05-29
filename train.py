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
#	Make model more accepting of different data (more channels, variable time points)	
#	Try inputting wavelet transform
#	Inline call of test function
#	DATA AUGMENTATION: from long timespans, take multiple chunks that are 1150 time points long
# 	Play with batch_size, d_model, nhead, dim_ffwd, dropout, optimizer


model_number = 4
learning_rate = 0.00002
validate_fraction = 100

normalize = False
bandpass_filter = False
augment = False

document = True
print_training_data = True

num_epochs = 8000
batch_size = 64

optimizer_name = 'adam'

subjects = [1, 2, 3, 5, 6, 7, 8]
excluded_y = []




print('Building Transformer Model...')
model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=8, dim_feedforward=32, dropout=0.2, n_classes=4)


def get_data(s, bandpass_filter, normalize, excluded_y, augment):
	
	print(f'Applying Bandpass Filter: {bandpass_filter}.')
	print(f'Normalizing: {normalize}.')
	print(f'Augmenting: {augment}.')
	
	X_tr, Y_tr, X_val, Y_val, X_te, Y_te = pro.get_broad_data(subject_list=s, bandpass_filter=bandpass_filter, excluded=excluded_y, augment=augment)


	print(f'Excluding Y categories {excluded_y}...')
	X_tr, Y_tr = pro.exclude_categories(X_tr, Y_tr, excluded_y)
	X_val, Y_val = pro.exclude_categories(X_val, Y_val, excluded_y)
	X_te, Y_te = pro.exclude_categories(X_te, Y_te, excluded_y)


	print('Input Shape: ', X_tr.shape)

	#X_tr, Y_tr = pro.shuffle(X_tr, Y_tr)
	#X_val, Y_val = pro.shuffle(X_val, Y_val)
	#X_te, Y_te = pro.shuffle(X_te, Y_te)

	print(X_tr.shape, X_val.shape, X_te.shape)

	if normalize:
		X_tr, X_val, X_te = pro.normalize_data(X_tr, X_val, X_te)

	print(X_tr.shape, X_val.shape, X_te.shape)
	
	return X_tr, Y_tr, X_val, Y_val, X_te, Y_te



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



def train(model, model_number, num_epochs, batch_size, optimizer_name, learning_rate, subjects, bandpass_filter, normalize, excluded_y, document=False, print_training_data=False, validate_fraction=100, save_charts=True):

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

	save = False

	if save:

		X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_data(subjects, bandpass_filter, normalize, excluded_y, augment)

		print('Creating Dataset...')
		
		torch.save(X_tr, 'X_tr.pt')
		torch.save(Y_tr, 'Y_tr.pt')
		torch.save(X_val, 'X_val.pt')
		torch.save(Y_val, 'Y_val.pt')
		torch.save(X_te, 'X_te.pt')
		torch.save(Y_te, 'Y_te.pt')

	X_tr = torch.load('X_tr.pt')
	Y_tr = torch.load('Y_tr.pt')
	X_val = torch.load('X_val.pt')
	Y_val = torch.load('Y_val.pt')
	X_te = torch.load('X_te.pt')
	Y_te = torch.load('Y_te.pt')

	X_val = X_val[:int(len(X_val) / 5)]
	Y_val = Y_val[:int(len(Y_val) / 5)]
	assert(len(X_val) == len(Y_val))

	X_tr, Y_tr, X_val, Y_val, X_te, Y_te = get_data(subjects, bandpass_filter, normalize, excluded_y, augment)


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


train(model, model_number, num_epochs, batch_size, optimizer_name, learning_rate, subjects, bandpass_filter, normalize, excluded_y, document=True, print_training_data=True, validate_fraction=100, save_charts=True)
train(model, 5, num_epochs, batch_size, optimizer_name, learning_rate, subjects, bandpass_filter, True, excluded_y, document=True, print_training_data=True, validate_fraction=100, save_charts=True)
#train(model, 6, num_epochs, batch_size, optimizer_name, learning_rate, subjects, bandpass_filter, True, excluded_y, document=True, print_training_data=True, validate_fraction=100, save_charts=True)
#train(model, 7, num_epochs, batch_size, optimizer_name, learning_rate, subjects, bandpass_filter, True, excluded_y, document=True, print_training_data=True, validate_fraction=100, save_charts=True)




#torch.save(model.state_dict(), 'models/models_' + str(model_number) + '/model_1.pth')

'''
model.eval()
with torch.no_grad():
	test_preds = model(X_te)
	test_loss = F.cross_entropy(test_preds, Y_te)
	test_percent = pro.percent_correct(test_preds, Y_te)

	print(f'Test Loss: {test_loss}')
	print(f'Test Accuracy: {test_percent}')

'''





