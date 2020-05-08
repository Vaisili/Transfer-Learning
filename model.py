import torch
import torch.nn as nn
import numpy as np
from torchvision import models

class ResNetFashion(nn.Module):
	def __init__(self, base="resnet50", num_classes=20, use_pretrained=True):
		super(ResNetFashion, self).__init__()
		if base=="inception":
			self.base = models.inception_v3(pretrained=use_pretrained)
		else:  #use resnet50 in all other cases
			self.base = models.resnet50(pretrained=use_pretrained)
		flatten = self.base.fc.in_features
		modules = list(self.base.children())[:-1]
		self.base = nn.Sequential(*modules)
		# we will not freeze any layers
		# add any layers in between base and output layer
		self.base.add_module('flatten', nn.Flatten())
		self.fc = nn.Linear(2048,num_classes)

	def forward(self, x):
		x = self.base(x)
		x = self.fc(x)
		return x