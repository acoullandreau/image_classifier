# This file contains tne script used to create and train a model on a dataset
# This script is written in Python 3
# To launch it the first time: python3 train.py flowers --arch 'vgg16' --epochs 2
# To launch it from checkpoint: python3 train.py flowers --checkpoint 'checkpoint.pth'

from model_classfile import MLModel, Classifier
from utility import Utils
import argparse
import torch

# We define the arguments the user can pass when launching the script
# We volontarily give the option to pass many parameters to launch the script
# Note: three different architectures are available with this version of the script (from model_classfile): vgg16, vgg19 and densenet121
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to images directory')
parser.add_argument('-m', '--arch', type=str, help='Pretrained model to load')
parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint file to load')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01 , help='Learning rate')
parser.add_argument('-l', '--layers', action='append', type=list, default=[256, 128, 102], help='Number of hidden and output layers')
parser.add_argument('-e', '--epochs', type=int, default = 50, help='Number of epochs')
parser.add_argument('-d', '--use_gpu', type=bool, default=True, help='Train model on GPU')
parser.add_argument('-s', '--save_dir', type=str, help='Directory where to save the checkpoint')

args = parser.parse_args()

data_dir = args.path
architecture = args.arch
learning_rate = args.learning_rate
hidden_layers = args.layers[0:-1]
output_layer = args.layers[-1]
epochs = args.epochs
save_dir = args.save_dir

# If the GPU is available and the user did not specify explicitely not to use it, we ensure it will be used
if args.gpu != None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if device == 'cpu':
    print('Running on CPU')

# We create our model
model = MLModel()
model.device = device

if args.checkpoint == None:
    if architecture != None:
        model.architecture = architecture
        if architecture == 'vgg16' or 'vgg19':
            model.input_size = 25088
        elif architecture == 'densenet121':
            model.input_size = 1024
    else:
        print('We could not build a model from a pretrained model - pretrained model unknown')
    model.hidden_size = hidden_layers
    model.output_size = output_layer
else:
    if device =='cpu':
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(args.checkpoint)
    if checkpoint != {}:
        model.architecture = checkpoint['architecture']
        model.input_size = checkpoint['input_size']
        model.hidden_size = checkpoint['hidden_size']
        model.output_size = checkpoint['output_size']
        model.state_dict = checkpoint['state_dict']
        model.optimizer_dict = checkpoint['optimizer']
        model.class_to_idx = checkpoint['class_to_idx']

model.learning_rate = learning_rate
model.epochs = epochs             

model.BuildNetwork()

# We load the data, as well as the class_to_idx dictionary
train_dataloaders, test_dataloaders, valid_dataloaders, class_to_idx = Utils.LoadData(data_dir, ['train', 'test', 'valid'])
model.class_to_idx = class_to_idx

# We train the model
model.TrainNetwork(train_dataloaders, test_dataloaders)

# We save the checkpoint
model.SaveModel()