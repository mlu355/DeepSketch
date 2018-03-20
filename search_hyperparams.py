"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
					help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
	"""Launch training of the model with a set of hyperparameters in parent_dir/job_name

	Args:
		model_dir: (string) directory containing config, weights and log
		data_dir: (string) directory containing the dataset
		params: (dict) containing hyperparameters
	"""
	# Create a new folder in parent_dir with unique_name "job_name"
	model_dir = os.path.join(parent_dir, job_name)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	# Write parameters in json file
	json_path = os.path.join(model_dir, 'params.json')
	params.save(json_path)

	# Launch training with this config
	cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
																				   data_dir=data_dir)
	print(cmd)
	check_call(cmd, shell=True)


if __name__ == "__main__":
	# Load the "reference" parameters from parent_dir json file
	args = parser.parse_args()
	json_path = os.path.join(args.parent_dir, 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	params = utils.Params(json_path)

	# Perform hypersearch over learning_rates, batch sizes, dropout
	learning_rates = [] #[1e-4, 1e-2, 1e-3]
	confusion_factors = [0.6, 0.7, 0.4, 1, 0.3, 0.2, 0.1, 0.5]
	dropouts = [0.9, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5]
	batch_sizes = [128, 256, 64, 32, 512]
	for learning_rate in learning_rates:
		params.learning_rate = learning_rate
		job_name = "learning_rate_{}".format(learning_rate)
		launch_training_job(args.parent_dir, args.data_dir, job_name, params) 

	params = utils.Params(json_path)
	for confusion_factor in confusion_factors:
		params.confusion_factor = confusion_factor
		job_name = "confusion_factor_{}".format(confusion_factor)
		launch_training_job(args.parent_dir, args.data_dir, job_name, params) 

	params = utils.Params(json_path)
	for dropout in dropouts:
		params.dropout = dropout
		job_name = "dropout_{}".format(dropout)
		launch_training_job(args.parent_dir, args.data_dir, job_name, params) 

	params = utils.Params(json_path)
	for batch_size in batch_sizes:
		params.batch_size = batch_size
		job_name = "batch_size_{}".format(batch_size)
		launch_training_job(args.parent_dir, args.data_dir, job_name, params) 


	#Perform hypersearch over one parameter
	#learning_rates = [1e-4, 1e-3, 1e-2]

	#for learning_rate in learning_rates:
		# Modify the relevant parameter in params
	 #   params.learning_rate = learning_rate

		# Launch job (name has to be unique)
	  #  job_name = "learning_rate_{}".format(learning_rate)
	   # launch_training_job(args.parent_dir, args.data_dir, job_name, params)
