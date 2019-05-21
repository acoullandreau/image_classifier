#This file regroups the methods used to load the data and preprocess images
from torchvision import datasets, transforms
import torch
import json
from PIL import Image
import numpy as np


class Utils:
    
    @staticmethod
    def LoadData(data_dir, folders):
        # We define the transforms used on each image of the provided directory
        # Transforms are different for training and testing/validation sets
        train_data_transforms = transforms.Compose([transforms.RandomRotation(25),
                                                    transforms.RandomResizedCrop(224), 
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                                         [0.229, 0.224, 0.225])])

        test_data_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224), 
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])

        # We load the datasets with ImageFolder
        # In case the data_dir contains only train and test, or train and validation, and not the three folders, we add an if statement to load the images accordingly
        for folder in folders:
            if folder == 'train':
                train_image_datasets = datasets.ImageFolder(data_dir + '/train', transform=train_data_transforms)
            elif folder == 'test':
                test_image_datasets = datasets.ImageFolder(data_dir + '/test', transform=test_data_transforms)
            elif folder == 'valid':
                valid_image_datasets = datasets.ImageFolder(data_dir + '/valid', transform=test_data_transforms)

        # We then define the dataloaders
        dataloaders = []
        if train_image_datasets != None:
            train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
            dataloaders.append(train_dataloaders)
        if test_image_datasets != None:
            test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)
            dataloaders.append(test_dataloaders)
        if valid_image_datasets != None:
            valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
            dataloaders.append(valid_dataloaders)
            
        dataloaders.append(train_image_datasets.class_to_idx)
        
        return dataloaders
            
    
    @staticmethod
    def MapCategories(cat_file):
        with open(cat_file, 'r') as f:
            cat_to_name = json.load(f)
        return cat_to_name
 
    @staticmethod
    def ImagePreprocessing(image):
        ##First, we get as an input the size of the image
        width, height = image.size
        
        ## We resize the image
        ## If the height is bigger than the width, then we assign the size 256 to the width
        ## In order to keep the ratio between the two dimensions, we use a ratio variable
        if height > width :
            ratio = int(height/width * 256)
            new_size = 256, ratio
            image.thumbnail(new_size)
        else:
            ratio = int(width/height *256)
            new_size = ratio, 256
            image.thumbnail(new_size)
        
        ## Then we crop from the center
        width, height = image.size
        left_margin = (width - 224)/2
        right_margin = left_margin + 224
        bottom_margin = (height - 224)/2
        top_margin = bottom_margin + 224

        image = image.crop((left_margin, bottom_margin, right_margin,    
                       top_margin))
    
        ## We convert the color channels to floats 0-1
        np_image = np.array(image)
        image = np_image/255

        ## We normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean)/std
    
        ## Transpose the color channel, with the color channel in the first position
        image = image.transpose((2, 0, 1))

        return image
    
    @staticmethod
    def Predict(image_path, model, topk=5):
        ## We process the raw image
        image = Image.open(image_path)
        processed_image = Utils.ImagePreprocessing(image)
        image = torch.FloatTensor(processed_image)
   
        ## We move the image to the GPU if available
        image = image.to(model.device)
        image.unsqueeze_(0)
    
        ## We only want to predict, so we can disable dropout
        model.model.eval()
    
        ## We apply our model on the image, and calculate the top probabilities
        log_output = model.model.forward(image)
        ps = torch.exp(log_output)
        top_ps, top_class = ps.topk(topk)

        ## Convert index to class
        idx_to_class = {}
        for key in model.class_to_idx.keys():
            idx_to_class[model.class_to_idx[key]] = key

        # Extract an array with the top labels
        ## Convert tensor to numpy array
        top_class = top_class.to('cpu')
        top_class = top_class.numpy()
        top_class = [idx_to_class[x] for x in top_class[0]]
    
        #Extract an array with the top probabilities
        top_ps = top_ps.to('cpu')
        top_ps = top_ps.detach().numpy()
        top_ps = top_ps[0]
    
        return top_ps, top_class