

from torchsummary import summary

def get_notes_filename(model):
	d = 'models/models_' + str(model) + '/'
	name = 'notes_' + str(model) + '.txt'

	return d + name

def create_notes_file(model_number, model, lr, normalize, bandpass_filter, num_epochs, batch_size, excluded):
	
	f = open(get_notes_filename(model_number), 'w')

	f.write(f'Model {model_number}:\n')
	f.write(f'Epochs: {num_epochs}\n')
	f.write(f'Batch Size: {batch_size}\n')
	f.write(f'Learning Rate: {lr}\n')
	f.write(f'Normalize: {normalize}\n')
	f.write(f'Bandpass Filter: {bandpass_filter}\n')
	f.write(f'Excluded Y: {excluded}\n')
	f.write(str(summary(model, input_size=(1, 22, 1125), verbose=0)))
	f.write('\n\n')
	f.close()



def update_file(model, epoch, tr_loss, tr_acc, val_loss, val_acc):

	f = open(get_notes_filename(model), 'a')
	f.write(f'Epoch: {epoch}\t Training Loss:\t {tr_loss}\t Training Accuracy:\t {tr_acc}\n')
	f.write(f'Epoch: {epoch}\t Valid. Loss:\t {val_loss}\t Validation Accuracy:\t {val_acc}\n')
	f.close()