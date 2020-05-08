import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from model import ResNetFashion
from data_loader import FashionDataset
from parameters import Args
import os


def save_model_fn(epoch, model, optimizer, name):
	

	#save the model
	state = {
	    'epoch': epoch,
	    'state_dict': model.state_dict(),
	    'optimizer': optimizer.state_dict(),
	}

	torch.save(state, name)
	print("Model- {} saved successfully".format(name))


def train_model():

	transforms_ = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
	data = FashionDataset(Args.data_dir, Args.train_csv, transform=transforms_)
	dataset_loader = torch.utils.data.DataLoader(dataset=data, batch_size=Args.batch_size, shuffle=False)
	cuda = torch.cuda.is_available()
	model = ResNetFashion(base=Args.base_model, num_classes=Args.num_classes)


	if cuda:
		model.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=Args.learning_rate, momentum=Args.momentum)  
	#optimizer = optim.Adam(model.parameters(), lr=0.001)

	if Args.checkpoint != None:
		cp = torch.load(Args.checkpoint)
		model.load_state_dict(cp['state_dict'])
		optimizer.load_state_dict(cp['optimizer'])
	
	

	for epoch in range(Args.epochs):
		total_loss = 0.0
		for i, (data, target) in enumerate(dataset_loader):
			
			if cuda:
				data, target = data.cuda(), target.cuda()
			optimizer.zero_grad()
			output = model(data)
		
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			print("Epoch: {} Minibatch:{} Loss: {}".format(epoch, i, loss.item()))
			if i % Args.avg_loss_batch == 0 and i != 0:  #print loss after every 100 mini batches
				print("Avg loss over last {} batches: {}".format(total_loss/Args.avg_loss_batch))
				total_loss= 0.0

		if save_model:
			name = os.path.join(Args.output_dir, "fashion-"+str(epoch)+".pth")
			save_model_fn(epoch, model, optimizer, name)


if __name__ == "__main__":
	train_model()