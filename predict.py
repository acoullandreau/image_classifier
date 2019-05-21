# This file contains tne script used to predict the category of an image and its probability
# This script is written in Python 3
################### To launch it : python3 predict.py flowers/valid/102/image_08002.jpg --checkpoint 'checkpoint.pth'

from utility import Utils
from model_classfile import MLModel
import argparse
import torch

# We define the arguments the user can pass when launching the script
# We volontarily give the option to pass many parameters to launch the script
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to image')
parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint file to load')
parser.add_argument('-k', '--top_k', type=int, default = 5, help='Top-k most likely classes')
parser.add_argument('-n', '--category_names', default='cat_to_name.json', type=str, help='Mapping of categories to real names') 
parser.add_argument('-d', '--gpu', type=bool, help='Train model on GPU')

args = parser.parse_args()

# We instantiate the model from a checkpoint
# If the GPU is available and the user did not specify explicitely not to use it, we ensure it will be used
if args.gpu != None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


if device.type == 'cpu':
    print('Running on CPU')

# We create our model
model = MLModel()
model.device = device

if args.checkpoint != None:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if checkpoint != {}:
        model.architecture = checkpoint['architecture']
        model.input_size = checkpoint['input_size']
        model.hidden_size = checkpoint['hidden_size']
        model.output_size = checkpoint['output_size']
        model.state_dict = checkpoint['state_dict']
        model.optimizer_dict = checkpoint['optimizer']
        model.class_to_idx = checkpoint['class_to_idx']
else:
    print('We could not predict the class of the image - trained model to use undefined. Please specify the model checkpoint filename.')
    
model.BuildNetwork()

# We store the other arguments to be used in the predict function
image_path = args.path
topk = args.top_k
if args.category_names != None:
    cat_to_name = Utils.MapCategories(args.category_names)
    
# We calculate the top probabilities and classes
top_ps, top_class = Utils.Predict(image_path, model, topk)

#For each class, we extract the corresponding name
top_class_names = []
for item in top_class:
    top_class_names.append(cat_to_name[item])

#We print the flower name and class probability
print("Based on the model prediction, this flower is a {}, with a probability of {}%.".format(top_class_names[0], top_ps[0]*100))

# We print the top_k classes names
if topk > 1:
    print('With lower probabilities, the flower could be')
    for k in range(1, topk):
        print('{}'.format(top_class_names[k]))
