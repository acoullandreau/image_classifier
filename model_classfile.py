# This file regroups the classes and methods related to the model creation, training, saving and loading

from torchvision import models
import torch
from torch import nn, optim
import torch.nn.functional as F


class MLModel:
    
    def __init__(self):
        self.architecture = None
        self.input_size = 25088
        self.hidden_size = [256, 128]
        self.output_size = 102
        self.learning_rate = 0.01
        self.optimizer_dict = None
        self.epochs = 50
        self.device = None
        self.class_to_idx = None
        self.state_dict = None
        self.parameters = None
        self.criterion = None
        self.model = None

    def BuildNetwork(self):
        if self.architecture == 'vgg16' or self.architecture == 'vgg19':
            model = models.vgg16(pretrained=True)
            self.input_size = 25088
        elif self.architecture == 'resnet18':
            model = models.densenet121(pretrained=True)
            self.input_size = 1024
        else:
            print('We could not build a model from a pretrained model - pretrained model unknown')
        
        ## Freeze the parameters, by turning off gradient descent (no backpropagation on the features layers, only on the classifier)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = Classifier(self.input_size, self.hidden_size, self.output_size)
        self.criterion = nn.NLLLoss()
    
        ## we load the state_dict
        model.load_state_dict(self.state_dict)

        ## we load the optimizer
        if self.optimizer_dict == None:
            optimizer = optim.SGD(model.classifier.parameters(), lr = self.learning_rate)
        else:
            optimizer = optim.SGD(model.classifier.parameters(), lr = self.learning_rate)
            optimizer.load_state_dict(self.optimizer_dict)
    
        ## we reuse the same class to index attribute
        model.class_to_idx = self.class_to_idx
    
        # We move our model to the GPU if available
        model.to(self.device)
        
        self.model = model
        
    
    def TrainNetwork(self, train_dataloaders, test_dataloaders):
        steps = 0
        for e in range(self.epochs):
            ## First we train the model on the train_dataloaders dataset
            training_loss = self.stage_model(train_dataloaders, 'train')

            ## Then we test our model on the testing set
            ## To increase the speed of this part of the code, we turn off dropout (as we do not need it for testing), using the model.eval() mode
            self.model.eval() 
            testing_loss = self.stage_model(test_dataloaders, 'test')
            accuracy = self.calculate_accuracy(test_dataloaders)

            self.model.train()

            ## At each epoch, we print the performance metrics
            print("Epoch: {}/{}.. ".format(e+1, self.epochs),
                  "Training Loss: {:.3f}.. ".format(training_loss/len(train_dataloaders)),
                  "Validation Loss: {:.3f}.. ".format(testing_loss/len(test_dataloaders)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(test_dataloaders)))  
        
            
    def stage_model(self, dataset, loop_type):
        stage_loss = 0
        for images, labels in dataset:
            ## We move the data to the GPU if available
            images, labels = images.to(self.device), labels.to(self.device)
            if loop_type == 'train':
                ## Set back the gradient to 0 for each new epoch
                self.optimizer.zero_grad()     
                ## Increment the stage_loss counter with the value of the loss at each epoch
                stage_loss += self.calc_loss(self.apply_model(images), labels)
                ## Use the optimizer to apply corrections on the weights and bias
                self.optimizer.step()
            else:
                stage_loss += self.calc_loss(self.apply_model(images), labels)

        return stage_loss
        
    def calculate_accuracy(self, dataset): 
        accuracy = 0
        for images, labels in dataset:
            ## We move the data to the GPU if available
            images, labels = images.to(self.device), labels.to(self.device)

            ## The output of the model is log, so to get probabilities we use the exponential function
            output = torch.exp(self.apply_model(images))

            ## We want to use only the top probability and top classes 
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            ## We use the mean of the distribution to determine the accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()            
        return accuracy

    def apply_model(self, image):
        ## Apply our model to the input images
        model_output = self.model(image)
        return model_output
  
    def calc_loss(self, output, label):
        ## Calculate the loss and apply backpropagation
        loss = self.criterion(output, label)
        loss.backward()
        return loss
    
    def SaveModel(self):
        if self.class_to_idx != None:
            checkpoint = {'architecture':self.architecture,
              'input_size': self.input_size,
              'hidden_size':self.hidden_size,
              'output_size':self.output_size,
              'state_dict': self.model.state_dict(),
              'optimizer':self.optimizer.state_dict(),
              'class_to_idx':self.class_to_idx}
        else:
            print('We could not save the model - missing class_to_idx attribute')
        torch.save(checkpoint, 'checkpoint.pth')
     
    
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0]) #hidden_layer_1 with same input as first layer of VGG classifier
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1]) #hidden_layer_2
        self.fc3 = nn.Linear(hidden_size[1], output_size) #output layer with 102 classes
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
