"""Evaluates the model"""
import argparse
import logging
import os
import platform

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SKETCHES', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

parser.add_argument('--file', default='/user_imgs/carrot.png', help="name of image to load and test")
SIZE = 128 

# creates an array mapping index to class name
def get_classes():
    cwd = os.getcwd()

    if platform.system() == 'Windows':
        data_path = os.path.join(cwd, 'data\\SKETCHES')
    else:
        data_path = os.path.join(cwd, 'data/SKETCHES')

    filenames= os.listdir(".") # get all files' and folders' names in the current directory

    # getting all the subfolders
    classes = []
    for f in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, f)): # check whether the current object is a folder or not
            classes.append(f)

    # print("class: ", len(classes))
    return classes

def setup_model(params, model_dir='experiments/base_model', restore_file='best'):
    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Define the model
    # use GPU if available
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, restore_file + '.pth.tar'), model)

    return model

def resize_and_save(filename, output_dir, new_name, size=SIZE):
        """Resize the image contained in `filename` and save it to the `output_dir`"""
        image = Image.open(filename)
        # Use bilinear interpolation instead of the default "nearest neighbor" method
        image = image.resize((size, size), Image.BILINEAR)
        image.save(os.path.join(output_dir, new_name))

def image_loader(params, image_name):
    # each image is 64x64
    loader = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

    """load image, returns cpu tensor turn it into grayscale"""
    image = Image.open(image_name).convert('L')

    # resize and turn image to tensor
    image = loader(image).float()
    #print("image shape: ", image.shape)
    image = Variable(image, requires_grad=True)

    # should have dims 1 x num_channels x height x width
    image = image.unsqueeze(0)  
    #print("image shape: ", image.shape)

    # use GPU if available
    
    if params.cuda:
        return image.cuda()
    else:
        return image.cpu() 

def classify_img(img_path, json_path='experiments/base_model/params.json'):
     # Load the parameters
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()

    print("oarams test: ", params.batch_size)

    # get classes
    classes = get_classes()

    # setup model
    model = setup_model(params) 

    # load test image
    image = image_loader(params, img_path)
    
    # set model to evaluation mode
    model.eval()
    output = model(image)
    label = np.argmax(output.cpu().data.numpy())

    print("predicted label: ", label)
    print("model says u drew a: ", classes[label])

    return label, classes[label]


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # get img file path
    cwd = os.getcwd()
    filename = args.file
    img_path = os.path.join(cwd, filename)

    classify_img(img_path)
    

    # # get classes
    # classes = get_classes()

    # # setup model
    # model = setup_model(params) 

    # # get image path
    # cwd = os.getcwd()

    # #curr_path = os.path.join(cwd, "data/64x64_SKETCHES/test_sketches/16_1315.png")
    # #curr_path = os.path.join(cwd, "data/64x64_SKETCHES/test_sketches/209_3721.png")

    # #filename = "apple.png"
    # filename = args.file
    # curr_path = os.path.join(cwd, filename)
    # #curr_path = cwd + filename
    # print(curr_path)

    # # not necessary
    
    # # new_dir = 'resized'
    # # new_dir = os.path.join(cwd, new_dir)
    # # new_path = os.path.join(new_dir, filename)
    # # resize_and_save(curr_path, new_dir, filename)

    # # load test image
    # image = image_loader(curr_path)
    
    # # set model to evaluation mode
    # model.eval()
    # output = model(image)
    # #print(output)
    # #predictions = output.max(1) 
    # #print("predictions", predictions)
    # label = np.argmax(output.cpu().data.numpy())
    # #print(most_probable)
    # # first element in predictions is the score, second element is the class index
    # #for i in range(len(classes)):
    # #    print(i, classes[i])    
    # #label = int(predictions[1])
    # print("predicted label: ", label)
    # print("model says u drew a: ", classes[label])
