import glob
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


#pytorch Dataset class to load provided data
class FashionDataset(Dataset):
	def __init__(self, data_dir, csv_path, transform=None):
		self.root_dir = data_dir
		self.image_names = glob.glob(data_dir+"*.jpg")
		self.image_names_base = [os.path.basename(i) for i in self.image_names]
		# print(self.image_names_base)
		self.txt_names = [i[:-4] + '.txt' for i in self.image_names_base]
		self.labels = torch.FloatTensor(self.generate_labels())
		# print(self.labels)
		self.labels_s = self.labels[:,0]
		self.labels_c = self.labels[:,1]
		self.labels_t = self.labels[:,2]
		self.transform = transform
		self.shape = shape

	def __getitem__(self, index):
		self.image = Image.open(self.image_names[index])
		#uncomment following line to read image in grayscale
		# self.image = Image.open(self.image_names[index]).convert('L')
		if self.shape == 's':
			self.label = self.labels_s[index]
		elif self.shape == 'c':
			self.label = self.labels_c[index]
		elif self.shape == 't':
			self.label = self.labels_t[index]
		else:
			self.label = self.labels[index]
		if self.transform:
			self.image = self.transform(self.image)
		return (self.image, self.label)

	def __len__(self):
		return len(self.image_names)



# for testing purpose

if __name__ == "__main__":
	data_loader = FashionDataset("training_data/", transform=transforms.Compose([transforms.Resize((280,280)), transforms.ToTensor()]))
	i = 0
	for img, label in data_loader:
		print(label)
		d = transforms.ToPILImage()(img)
		i += 1
		if i == 134:
			d.show()
			break
		# break