#######
# parameters for training.
#######

import os

class Args:
	# give a name to this task or a set of parameters
	# models will be saved in this folder
	name = "remainingclasses_loss_weight_sgd"
	batch_size = 16
	data_dir = "D:\\Projects\\Datasets\\fashion-larger"
	train_csv = "remainingclasses_set.csv"
	learning_rate = 0.003
	momentum = 0.9
	epochs = 10
	# checkpoint to start training from
	# checkpoint = "D:\\Projects\\Datasets\\fashion-larger\\model_output\\fashion-9.pth"
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
	# if True then only throws warning whenever possible
	be_nice = False

	def __init__(self):
		self.verify_args()

	def verify_args(self):
		if not os.path.exists(self.data_dir):
			raise ValueError("Invalid path for data_dir. Directory does not exists!")
		if not os.path.exists(os.path.join( self.output_dir, self.name)):
			os.makedirs(os.path.join(Args.output_dir, self.name))
			self.save_settings()
		else:
			if not any(fname.endswith('.pth') for fname in os.listdir(os.path.join(self.output_dir, self.name))):
				self.save_settings()
				return
			while True:
				user_choice = input("Directory with the choosen task name already exists. Do you want to use checkpoint from this directory (y/n)?")
				if user_choice == "y":
					#overwrite checkpoint with the one in this dir
					self.checkpoint = self.find_latest_checkpoint()
					break
				elif user_choice == "n":
					break
				else:
					print("please enter a valid option.(y/n)")


	def save_settings(self):
		with open(os.path.join(Args.output_dir, self.name, "config.txt"), "w") as cfg:
			cfg.write("####################################################\n")
			cfg.write("#             Settings for Task: {}            #\n\n".format(Args.name))
			for var in dir(Args):
				if not var.startswith("__") and not callable(getattr(Args, var)):
					cfg.write("{} : {} \n".format(var, getattr(Args,var)))



_ = Args()