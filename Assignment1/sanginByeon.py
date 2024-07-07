# Question 1. How can I move to the directory I want using the library os & Linux Command?
# I'd like to move to the directory name '/content/sample_data'.

import os

os.getcwd()
os.chdir('/content/sample_data')


# Question 2. Please write a code that copies new.txt with a file name new3.txt.
# Directory: /content/drive/MyDrive/intro-dl/afhq/new_folder

import shutil

src3 = '/content/drive/MyDrive/제목없는 폴더/afhq/new_folder/new.txt'
dst3 = '/content/drive/MyDrive/제목없는 폴더/afhq/new_folder/new3.txt'


shutil.copy(src3, dst3)


# Question 3. Compute L1/L2 Norm between matrix1 and matrix2 above.

import torch

matrix1 = torch.tensor([[1., 2.], [3., 4.]])
matrix2 = torch.tensor([[5., 6.], [7., 8.]])

## L1 Norm
norm1 = torch.linalg.matrix_norm(matrix1-matrix2, ord=1)
## L2 Norm
norm2 = torch.linalg.matrix_norm(matrix1-matrix2, ord=2)


# Question 4. Please write a line-by-line explanation of the code above. (Simple MLP only)

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


## Checks if CUDA (GPU) is available and sets the device to GPU if available, otherwise sets it to CPU.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

## Prints the version of PyTorch being used and the device (CPU or GPU) selected.
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)


## Defines the batch size for loading data and the number of epochs for training the model.
BATCH_SIZE = 32
EPOCHS = 10


## Downloads the MNIST training dataset and applies a transformation to convert the images to tensors.
''' 3. MNIST Download (Train set, Test set split) '''
train_dataset = datasets.MNIST(root = "../data/MNIST", train = True, download = True, transform = transforms.ToTensor())

## Downloads the MNIST test dataset and applies the same transformation to convert the images to tensors.
test_dataset = datasets.MNIST(root = "../data/MNIST", train = False, transform = transforms.ToTensor())

## Creates a DataLoader for the training dataset with the specified batch size and shuffling enabled.
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)

## Creates a DataLoader for the test dataset with the specified batch size and shuffling disabled.
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


## Iterates over one batch of the training data, prints the size and type of the input images and labels, then breaks the loop.
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

## Plots the first 10 images of the batch with their respective class labels.
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
    plt.title('Class: ' + str(y_train[i].item()))

## Defines a Multi-Layer Perceptron (MLP) class Net with three fully connected layers.
    ''' Multi Layer Perceptron '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)


## Defines the forward pass of the network. Reshapes the input, applies the first layer with a sigmoid activation function,
## applies the second layer with a sigmoid activation function, applies the third layer, and then applies the log softmax function.
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x
    

## Creates an instance of the Net class and moves it to the specified device (CPU or GPU).
''' Optimizer, Objective Function '''
model = Net().to(DEVICE)

## Initializes the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.5. Defines the loss function as Cross-Entropy Loss.
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()


## Defines the training function. 
# Sets the model to training mode
# iterates over batches, moves images and labels to the specified device, 
# performs forward and backward passes, updates weights, and prints the training progress at specified intervals.
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            

## Defines the evaluation function. Sets the model to evaluation mode, calculates the test loss and accuracy, and returns these metrics.
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


## Trains and evaluates the model for the specified number of epochs, printing the test loss and accuracy at the end of each epoch.
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))

# Question 5 (Optional). Please read and summarize the following 3 documents:
