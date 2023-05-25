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
# 	Try choosing between only 2 options
#	Try inputting wavelet transform

model_number = 10
learning_rate = 0.00003
validate_fraction = 100
normalize = True
bandpass_filter = False
document = True

print('Building Transformer Model...')
model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=32, dropout=0.2, n_classes=2)

s = [1, 2, 3, 5, 6, 7, 8]
print('Assembling Dataset...')
print(f'Applying Bandpass Filter: {bandpass_filter}.')
print(f'Normalizing: {normalize}.')

excluded = [2, 3]
X_tr, Y_tr, X_val, Y_val, X_te, Y_te = pro.get_broad_data(subject_list=s, bandpass_filter=bandpass_filter, normalize=normalize, excluded=excluded)


print(f'Excluding Y categories {excluded}...')
#X_tr, Y_tr = pro.exclude_categories(X_tr, Y_tr, excluded)
#X_val, Y_val = pro.exclude_categories(X_val, Y_val, excluded)
#X_te, Y_te = pro.exclude_categories(X_te, Y_te, excluded)


#X_tr, Y_tr, X_val, Y_val, X_te, Y_te = pro.get_split_data(subject_id=1)
print('Input Shape: ', X_tr.shape)

#input_data = torch.randn((32, 1, 22, 1125))
num_epochs = 9000
batch_size = 32

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


if document: doc.create_notes_file(model_number, model, learning_rate, normalize, bandpass_filter, num_epochs, batch_size, excluded)

print('Training...')

def save_losses(epoch):

	plt.figure()
	plt.plot(epochs, val_losses, label='val loss', color='blue')
	plt.plot(epochs, tr_losses, label='train loss', color='red')
	plt.title('Validation Losses, Epoch ' + str(epoch))
	plt.xlabel('Epoch')
	plt.legend()
	plt.savefig('charts/charts_' + str(model_number) + '/losses_' + str(epoch) + '.png')
	plt.figure()
	plt.close('all')

def save_accuracies(epoch):
	plt.figure()
	plt.plot(epochs, val_accuracies, label='val accuracy', color='blue')
	plt.plot(epochs, tr_accuracies, label='train accuracy',color='red')
	plt.legend()
	plt.title('Validation Percent Correct, Epoch ' + str(epoch))
	plt.xlabel('Epoch')
	plt.savefig('charts/charts_' + str(model_number) + '/accuracies_' + str(epoch) + '.png')
	plt.figure()
	plt.close('all')



def train(learning_rate):

	
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	

	# X_tr (218, 1, 22, 1125)
	# Y_tr (218, 4)

	for epoch in range(num_epochs):

		
		
		if epoch == 0 or (epoch + 1) % (int(num_epochs / validate_fraction)) == 0:
			print(f"[Epoch {epoch + 1} / {num_epochs}]")

			model.eval()
			with torch.no_grad():

				ix = torch.randint(0, X_tr.shape[0], (batch_size*3, ))
				Xb, Yb = X_tr[ix], Y_tr[ix]

				train_preds = model(Xb)

				train_loss = F.cross_entropy(train_preds, Yb)
				train_percent = pro.percent_correct(train_preds, Yb)

				val_preds = model(X_val)
				val_loss = F.cross_entropy(val_preds, Y_val)
				val_percent = pro.percent_correct(val_preds, Y_val)

				print(f'Training Loss: {train_loss}')
				print(f'Training Accuracy: {train_percent}')
				print(f'Validation Loss: {val_loss}')
				print(f'Validation Accuracy: {val_percent}')
				
				tr_losses.append(train_loss)
				tr_accuracies.append(train_percent)
				val_losses.append(val_loss)
				val_accuracies.append(val_percent)
				epochs.append(epoch+1)

			model.train()
		
		if (epoch + 1) % 1000 == 0:
			torch.save(model.state_dict(), 'models/models_' + str(model_number) + '/model_' + str(model_number) + '_epoch_' + str(epoch+1) + '.pth')

			save_losses(epoch+1)
			save_accuracies(epoch+1)
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


train(learning_rate)
torch.save(model.state_dict(), 'models/models_' + str(model_number) + '/model_1.pth')

model.eval()
with torch.no_grad():
	test_preds = model(X_te)
	test_loss = F.cross_entropy(test_preds, Y_te)
	test_percent = pro.percent_correct(test_preds, Y_te)

	print(f'Test Loss: {test_loss}')
	print(f'Test Accuracy: {test_percent}')
	





