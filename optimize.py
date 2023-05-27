

from train import train
from model import ContinuousTransformer
import processing as pro
from test import get_max_accuracy

import torch
import torch.nn.functional as F


'''
Best parameters:

batch_size: 8
n_heads: 16
dim_ffwd: 64
dropout: 0.2
optimizer: adam

'''
f = open('optimize_2.txt', 'w')

batch_sizes = [16, 32, 64]
n_heads = [8, 16]
#d_models = [8, 16, 32]
dim_feedforwards = [32, 64, 128]
dropouts = [0.1, 0.2, 0.3]
optimizers = ['adam']


combined = [(bs, nh, dffwd, do, opt) 
	for bs in batch_sizes 
	for nh in n_heads
	#for dm in d_models
	for dffwd in dim_feedforwards
	for do in dropouts
	for opt in optimizers
]

losses = []
percents = []

s = [1, 2, 3, 5, 6, 7, 8]
excluded = []
lr = 0.00002
model_number = 3
num_epochs = 10000


f.write('Start\n')
f.close()
for c in combined:
	f = open('optimize_2.txt', 'a')
	bs, nh, dffwd, do, opt = c
	model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=nh, dim_feedforward=dffwd, dropout=do, n_classes=4)

	model = train(model, model_number, num_epochs, bs, opt, lr, s, normalize=True, bandpass_filter=False, excluded_y=excluded, document=False, print_training_data=False)

	min_loss, max_accuracy = get_max_accuracy(model, model_number=3, subject=9)

	f.write(f'Parameters: {c}\t Min Loss: {min_loss}\t Max Accuracy: {max_accuracy}.\n')
	f.close()

