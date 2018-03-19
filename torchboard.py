import os
import platform
import re
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

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
	for f in os.listdir(curr_path):
		if f == 'train.log':
			log = os.path.join(curr_path, f) 

	# split by 2018
	# re.compile("2018")
	train_loss = []
	eval_loss = []
	epochs = []
	f = open(log, 'r').read().split()

	# read file into string
	file = open(log, 'r').read()
	# print(re.split("2018-", f))
	print(file)
	# print(f)

	# get index of lines with epochs
	for j in range(0, len(f)):
		if f[j] == "Epoch":
			# num = f[j+1].split("/")
			epochs.append(j)

	# extract loss
	for j in range(0, len(epochs)):
		# epoch index +1 = train
		# epoch index +2 = eval
		train_loss.append(f[epochs[j]+11])
		print(f[epochs[j]+11])
		eval_loss.append(f[epochs[j]+22])

	print(train_loss)
	print(len(eval_loss))

	print(epochs)
	print(train_loss)


	plt.plot(np.arange(11)[1:], train_loss[:10], 'ro')
	# plt.axis([0, 6, 0, 20])
	plt.show()
	plt.savefig("testlol.png")

			

