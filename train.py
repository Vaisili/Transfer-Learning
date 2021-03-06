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
from torch.utils.tensorboard import SummaryWriter



def save_model_fn(epoch, model, optimizer, name, label_map):
	

	#save the model
	state = {
	    'epoch': epoch,
	    'state_dict': model.state_dict(),
	    'optimizer': optimizer.state_dict(),
	    'label_map':label_map,
	}

	torch.save(state, name)
	print("Model- {} saved successfully".format(name))


def get_class_weights(label_map, class_distribution, scheme):
	print("Calculating weights for classes...")
	# print(class_distribution)
	if scheme == 1:
		weights = [1/class_distribution[i] for i in label_map]
	elif scheme == 2:
		max_dist = max(class_distribution.values())
		weights = [max_dist/class_distribution[i] for i in label_map]
	# print(weights)
	return weights

def oversample(data):
	indices = list(range(len(data)))

	new_map = {value:key for (key,value) in data.label_map.items()}
	# print(data.class_distribution)
	# print([data.class_distribution[new_map[data[i][1]]] for i in indices][:10])
	weights = [1.0/data.class_distribution[new_map[data[i][1]]] for i in indices]
	# print(weights[:10])
	return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

def train_model():

	transforms_ = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
	data_cls = FashionDataset(Args.data_dir, Args.train_csv, transform=transforms_)
	if Args.oversample:
		print("Oversampling minority classes, this might take a while...")
		sampler = oversample(data_cls)
		dataset_loader = torch.utils.data.DataLoader(dataset=data_cls, batch_size=Args.batch_size, sampler=sampler)
	else:
		dataset_loader = torch.utils.data.DataLoader(dataset=data_cls, batch_size=Args.batch_size, shuffle=True)
	cuda = torch.cuda.is_available()
	model = ResNetFashion(base=Args.base_model, num_classes=Args.num_classes)


	if cuda:
		model.cuda()

	
	if Args.optimizer == "sgd":
		optimizer = optim.SGD(model.parameters(), lr=Args.learning_rate, momentum=Args.momentum)  
	elif Args.optimizer == "adam":
		optimizer = optim.Adam(model.parameters(), lr=Args.learning_rate)

	if Args.checkpoint != None:
		print("restoring from checkpoint ", Args.checkpoint)
		cp = torch.load(Args.checkpoint)
		# for n, v in cp['state_dict'].items():
			# print(n)
		if Args.diff_num_class:
			print("Restoring weights except last layer")
			model_dict = model.state_dict()
			pretrained_dict = cp['state_dict']
			# remove last two elements (fc weight and fc bias)
			# filtered_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}
			pretrained_dict.popitem()
			pretrained_dict.popitem()
			model_dict.update(pretrained_dict)
			model.load_state_dict(model_dict)
		else:
			print("Restoring weights for all layers")
			model.load_state_dict(cp['state_dict'])
			optimizer.load_state_dict(cp['optimizer'])
		# if Args.num_class_finetune != None:
		# 	num_features = model.fc.in_features
		# 	model.fc = nn.Linear(num_features, Args.num_class_finetune).cuda()

	
	if Args.weighted_loss and not Args.oversample:
		weights = get_class_weights(data_cls.label_map, data_cls.class_distribution, Args.weighted_loss_scheme)
		class_weights = torch.FloatTensor(weights)
		# if cuda:
			# class_weight.to('cuda')
		criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
	else:
		criterion = nn.CrossEntropyLoss()

	writer = SummaryWriter(os.path.join(Args.output_dir, Args.name))


	for epoch in range(Args.epochs):
		total_loss = 0.0
		correct = 0 
		for i, (data, target) in enumerate(dataset_loader):
			
			if cuda:
				data, target = data.cuda(), target.cuda()
			optimizer.zero_grad()
			output = model(data)
		
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			output_idx = output.argmax(dim=1, keepdim=True)
			correct += output_idx.eq(target.view_as(output_idx)).sum().item()
			print("Epoch: {} Minibatch:{} Loss: {}".format(epoch, i, loss.item()))
			# if i % Args.avg_loss_batch == 0 and i != 0:  #print loss after every 100 mini batches
			# 	print("Avg loss over last {} batches: {}".format(Args.avg_loss_batch, total_loss/Args.avg_loss_batch))
			# 	total_loss= 0.0
		avg_loss = total_loss/len(data_cls)
		accuracy = correct / len(data_cls)
		print("Average Loss Over {} Epochs: {}".format(epoch, avg_loss))
		print("After Epoch: {}  Accuracy: {}".format(epoch, accuracy))
		if Args.tensorboard:
			writer.add_scalar("Loss/train", avg_loss, epoch)
			writer.add_scalar("Accuracy/train", accuracy)

		if Args.save_model!= None and epoch % Args.save_model==0:
			name = os.path.join(Args.output_dir, Args.name, "fashion-"+str(epoch)+".pth")
			save_model_fn(epoch, model, optimizer, name, data_cls.label_map)


if __name__ == "__main__":
	train_model()