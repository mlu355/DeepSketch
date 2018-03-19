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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SKETCHES', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    # test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    # save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    # utils.save_dict_to_json(test_metrics, save_path)

    def image_loader(image_name):
        # each image is 64x64
        imsize = 64 
        loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])

        """load image, returns cpu tensor turn it into grayscale"""
        image = Image.open(image_name).convert('L')

        # resize and turn image to tensor
        image = loader(image).float()
        print("image shape: ", image.shape)
        image = Variable(image, requires_grad=True)

        # should have dims 1 x num_channels x height x width
        image = image.unsqueeze(0)  
        print("image shape: ", image.shape)

        return image.cpu()  

    # get image path
    cwd = os.getcwd()

    curr_path = os.path.join(cwd, "my_test.png")

    image = image_loader(curr_path)
    
    # set model to evaluation mode
    model.eval()
    output = model(image)
    predictions = output.max(1) 
    # label = res.data.cpu().numpy()
    print(predictions)
