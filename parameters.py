#######
# parameters for training.
#######

import os

class Args:
	batch_size = 16
	data_dir = "D:\\Projects\\Datasets\\fashion-larger"
	train_csv = "remainingclasses_set.csv"
	learning_rate = 0.003
	momentum = 0.9
	epochs = 10
	# checkpoint to start training from
	# checkpoint = "D:\\Projects\\Datasets\\fashion-larger\\model_output\\fashion-1.pth"
	checkpoint = None
	# save model after every X epochs
	save_model = 1
	output_dir = "D:\\Projects\\Datasets\\fashion-larger\\model_output"
	base_model = "resnet50"
	num_classes = 88  # top 20 classes
	# calculate avg loss for X minibatches
	avg_loss_batch = 100
	oversample=False    #over sample to handle class imbalance
	weighted_loss = True   # oversampling takes priority over weighted loss.
	# weighted loss schemes
	# 1. weight = 1/class_size 
	# 2. max(class frequency)/freq of given class
	weighted_loss_scheme = 2

	def __init__(self):
		self.verify_args()

	def verify_args(self):
		if not os.path.exists(self.data_dir):
			raise ValueError("Invalid path for data_dir. Directory does not exists!")
		if not os.path.exists(self.output_dir):
			os.makedirs(Args.output_dir)

_ = Args()