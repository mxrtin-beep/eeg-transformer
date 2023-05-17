import torch
import torch.nn.functional as F

import processing as pro
from model import ContinuousTransformer



model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=512, dropout=0.1)

def load_model(model_number, model):
	filename = 'models/models_' + str(model_number) + '/model_' + str(model_number) + '.pth'
	model.load_state_dict(torch.load(filename))
	#optimizer.load_state_dict(checkpoint["optimizer"])

	return model

def test(model, model_number, subject):

	print(f'Testing Model {model_number} on Subject {subject}')
	X, Y = pro.get_subject_dataset(subject)
	

	model = load_model(6, model)


	model.eval()
	with torch.no_grad():
		preds = model(X)
		#print(preds)
		#print(Y)
		loss = F.cross_entropy(preds, Y)
		
	percent = pro.percent_correct(preds, Y)

	print(f'Test Loss: {loss}')
	print(f'Percent Correct: {percent}')

	#pro.print_predictions(preds, Y)
		#print('Preds ', val_preds[:10])
		#print('Targets ', Y_val[:10])


MODEL_N = 6
N_SUBJECTS = 9

# 4: has multiple cues per start_session
SKIP_SUBJECTS = {4}

for i in range(1, N_SUBJECTS+1):
	if i not in SKIP_SUBJECTS:
		test(model, MODEL_N, i)

#test(model, MODEL_N, 4)








