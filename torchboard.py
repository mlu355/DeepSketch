import os
import platform
import re
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("ticks")

# graph loss over time
# parsing log file to extract loss data
learning_rate = ['learning_rate_0.0001', 'learning_rate_0.001', 'learning_rate_0.01']

if __name__ == '__main__':
	# find the log file
	cwd = os.getcwd()

	if platform.system() == 'Windows':
		dst = os.path.join(cwd, 'experiments\\learning_rate')
	else:
		dst = os.path.join(cwd, 'experiments/learning_rate')

	curr_path = os.path.join(dst, learning_rate[0])

	# getting ther log file
	# for f in os.listdir(curr_path):
	# 	if f == 'train.log':
	# 		log = os.path.join(curr_path, f) 

	# getting text.log file since gitignore scerewed up actual log files
	for f in os.listdir(cwd):
		if f == "log.txt":
			log = os.path.join(cwd, f)

	
	epochs_indices = []
	f_str = open(log, 'r').read()

	# cleans file to extract losses more easily
	cleaned_file = re.sub(r'([0-9]){4}-([0-9]){2}-([0-9]){2} ([0-9]){2}:([0-9]){2}:([0-9]){2},([0-9]){3}:INFO: | [:;]', '', f_str)
	f = cleaned_file.split()
	# print(cleaned_file)

	num_epochs = 0
	# get index of lines with epochs
	for j in range(0, len(f)):
		if f[j] == "Epoch":
			# num = f[j+1].split("/")[1]
			epochs_indices.append(j)
			# get number of epochs
			if num_epochs == 0:
				num_epochs = int(f[j+1].split("/")[1])

	# print("here are ", len(epochs_indices))
	train_loss = [0 for i in range(num_epochs+1)]
	eval_loss = [0 for i in range(num_epochs+1)]

	# extract loss
	for j in range(0, len(epochs_indices)):

		# because sometimes the logs are out of order...
		curr_epoch = int(f[epochs_indices[j]+1].split("/")[0])
		# print(curr_epoch)

		# find Train metrics:
		upper_bound = len(f)-5
		if j+1 < len(epochs_indices):
			upper_bound = epochs_indices[j+1]

		# log file is a bit jumbled...
		num_train = 0
		for k in range(epochs_indices[j], upper_bound):
			if f[k] == "Train":
				# first train belonged to the previous epoch which appeared in the log (also not in chronological epoch order RIP WTF)
				prev_epoch = int(f[epochs_indices[j-1]+1].split("/")[0])
				
				if num_train == 1 and prev_epoch >= 1:
					train_loss[prev_epoch] = first_train_loss
					# print(prev_epoch)

				# if there are more than one train losses in a epoch, the first one actually belongs to the previous epoch
				first_train_loss = float(f[k+5])
				train_loss[curr_epoch] = float(f[k+5])
				num_train += 1

			# strangely, eval losses are not scrambled like the train losses
			if f[k] == "Eval":
				eval_loss[curr_epoch] = float(f[k+5])

	# test to see epochs map correctly to loss
	for j in range(1, num_epochs+1):
		print("Epoch: ", j, " train loss: ", eval_loss[j])

	print(max(train_loss))
	print(max(eval_loss))

	# train plot
	plt.figure(0)
	plt.plot(np.arange(num_epochs+1)[1:], train_loss[1:], 'ro') # since train epoch indexing starts at 1
	plt.yticks(np.arange(min(train_loss), max(train_loss)+0.1, 0.1))
	plt.savefig("train.png")

	# eval plot
	plt.figure(1)
	plt.plot(np.arange(num_epochs+1)[1:], eval_loss[1:], 'ro') # since train epoch indexing starts at 1
	plt.yticks(np.arange(min(eval_loss), max(eval_loss), 1))
	plt.show()
	plt.savefig("eval.png")

	# together
	plt.figure(2)
	plt.plot(np.arange(num_epochs+1)[1:], train_loss[1:], 'r--', np.arange(num_epochs+1)[1:], eval_loss[1:], 'bs')
	plt.yticks(np.arange(min(min(eval_loss), min(train_loss)), max(max(eval_loss), max(train_loss)), 1))
	plt.legend()
	plt.show()
	plt.savefig("together.png")

			

