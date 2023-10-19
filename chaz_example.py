#Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision  import datasets, transforms
import torchvision

#Data
cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
#Loaders
train_loader = DataLoader(cifar_train, batch_size = 128, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = 128, shuffle=True)


#Set up class
class Model_1(nn.Module):
    def __init__(self, input, output):
        super(Model_1, self).__init__()
        #(Input_size Output_size)
        self.L1 = nn.Linear(input, 100)
        self.L2 = nn.Linear(100, 100)
        self.L3 = nn.Linear(100, output)
        self.Relu = nn.ReLU()
    def forward(self, x):
        layer_1 = self.Relu(self.L1(x))
        layer_2 = self.Relu(self.L2(layer_1))
        output = self.L3(layer_2)
        return output
'''
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.Model = nn.Sequential(
            #(Input_size Output_size)
            nn.Linear(10, 100)
            nn.ReLU()
            nn.Linear(100, 100)
            nn.ReLU()
            nn.Linear(100, 5)
        )
    def forward(self, x):
        output = self.Model(x)
        return output
'''    
#Input --> Imsge Flattened (32*32*3)
#Output --> The number of classes (10)
Model_1 = Model_1(3072, 10)

opt = optim.Adam(Model_1.parameters(), lr = 0.0001)

def epoch(model, data_loader, opt=None):
    total_loss = 0.
    total_acc = 0.
    #Loop through each bach
    for X, y in data_loader:
        #Flatten Images
        X = X.reshape(X.size(0), -1)
        #Get Model output
        Class_prediction = model(X)
        #Get Loss
        #Cross Entropy (includes Softmax activation on output layer)
        Loss = nn.CrossEntropyLoss()(Class_prediction, y)
        #Optimize if training
        if opt:
            opt.zero_grad()
            Loss.backward()
            opt.step()
        #Total loss update
        total_loss += Loss.item() * X.shape[0]
        #Total Accuracy Update
        total_acc += (Class_prediction.max(dim=1)[1] == y).sum().item()
    return total_loss / len(data_loader.dataset), total_acc / len(data_loader.dataset)
    
epochs = 3
#Training
for i in range(epochs):
    print(f'Epoch {i+1}')
    print('----------------------')
    training_loss, training_accuracy = epoch(Model_1, train_loader, opt)
    print(f'Training Loss: {training_loss}')
    print(f'Training Accuracy: {training_accuracy}')
    print('----------------------')
    testing_loss, testing_accuracy = epoch(Model_1, test_loader)
    print(f'Testing Loss: {testing_loss}')
    print(f'Testing Accuracy: {testing_accuracy}')
    print('----------------------')
