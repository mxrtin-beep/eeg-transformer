
from model import ContinuousTransformer
import torch
from torchtext.data import Field, BucketIterator



train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)



model = ContinuousTransformer(in_channels=1, out_channels=8, d_model=16, nhead=4, dim_feedforward=512, dropout=0.1)
input_data = torch.randn((10, 1, 1, 22, 1125))

num_epochs = 10
out = model(input_data)

print('Out: ', out.shape)
print(out)



for epoch in range(num_epochs):

	print(f"[Epoch {epoch} / {num_epochs}]")

	model.train()