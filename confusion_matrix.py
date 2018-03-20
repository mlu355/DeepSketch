"""Evaluates the model"""
import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

from PIL import Image
import torchvision.transforms as transforms

# confusion matrix
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SKETCHES', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

parser.add_argument('--file', default='user_imgs/carrot.png', help="name of image to load and test")
SIZE = 64


# creates an array mapping index to class name
def get_classes():
    cwd = os.getcwd()

    data_path = os.path.join(cwd, 'data/SKETCHES')
    filenames= os.listdir(".") # get all files' and folders' names in the current directory

    # getting all the subfolders
    classes = []
    for f in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, f)): # check whether the current object is a folder or not
            classes.append(f)

    # print("class: ", len(classes))
    return classes

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    class_true = []
    class_pred = []
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # get label from each output in batch
        for i in range(output_batch.shape[0]):
            predicted_label = np.argmax(output_batch[i], axis=0)
            class_pred.append(predicted_label)
            class_true.append(labels_batch[i])

    # print("output_batch: ", class_pred)
    # print("labels_batch: ", class_true)()

    # store confusion matrix in file bc it takes forever to compute
    conf_mat = confusion_matrix(class_true, class_pred)
    np.save("conf_matrix", conf_mat)
    return conf_mat

def make_confusion_plot(arr):
    norm_conf = []
    for i in arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'A'*250
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', format='png')

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # get classes
    classes = get_classes()

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # load test dataset
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']


    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    # conf_matrix = evaluate(model, loss_fn, test_dl, metrics, params)
    conf_matrix = np.load("conf_matrix.npy")
    print("confused matrix: ", conf_matrix)
    # make_confusion_plot(conf_matrix[:10][:10])

    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')



