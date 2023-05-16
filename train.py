import torch
from torchtext.data import Field, BucketIterator
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import processing as pro
from model import ContinuousTransformer


learning_rate = 0.0001
validate_fraction = 100


model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=512, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#input_data = torch.randn((32, 1, 22, 1125))
num_epochs = 10000
batch_size = 32

val_losses = []
val_accuracies = []


def train():

	X_tr, Y_tr, X_val, Y_val, X_te, Y_te = pro.get_split_data(subject_id=1)

	

	# X_tr (218, 1, 22, 1125)
	# Y_tr (218, 4)

	for epoch in range(num_epochs):

		
		if epoch % (int(num_epochs / validate_fraction)) == 0:
			print(f"[Epoch {epoch} / {num_epochs}]")

			model.eval()
			with torch.no_grad():
				val_preds = model(X_val)
				val_loss = F.cross_entropy(val_preds, Y_val)
				val_percent = pro.percent_correct(val_preds, Y_val)

				print(f'Validation Loss: {val_loss}')
				print(f'Percent Correct: {val_percent}')
				#print('Preds ', val_preds[:10])
				#print('Targets ', Y_val[:10])
				val_losses.append(val_loss)
				val_accuracies.append(val_percent)

			model.train()
		
		if epoch % 1000 == 0:
			torch.save(model.state_dict(), 'model_1_epoch_' + str(epoch) + '.pth')
		
		ix = torch.randint(0, X_tr.shape[0], (batch_size, ))
		Xb, Yb = X_tr[ix], Y_tr[ix]
		#print(Xb.dtype)

		preds = model(Xb)
		loss = F.cross_entropy(preds, Yb)

		#print(f'Loss: {loss}')
		loss.backward()

		optimizer.step()
		#print(out.shape)


train()

plt.plot(val_losses)
plt.title('Validation Losses')
plt.xlabel('Epoch')
plt.savefig('Losses_1.png')

plt.figure()

plt.plot(val_accuracies)
plt.title('Validation Percent Correct')
plt.xlabel('Epoch')
plt.savefig('Accuracies.png')
