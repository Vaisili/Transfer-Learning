#######
# parameters for training.
#######
class Args:
	batch_size = 2
	data_dir = "D:\\Projects\\Datasets\\fashion-larger"
	train_csv = "top20classes_set.csv"
	learning_rate = 0.0003
	momentum = 0.9
	epochs = 2
	# checkpoint to start training from
	checkpoint = None
	# save model after every X epochs
	save_model = 1
	output_dir = "D:\\Projects\\Dataset\\fashion-larger\\model_output"
	base_model = "resnet50"
	num_classes = 20  # top 20 classes
	