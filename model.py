import torch.nn as nn
import torch
from torchsummary import summary
class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        # write your codes here
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)#(should be gaussian connection but don't know how to implement)
        self.activation = nn.ReLU()
    def forward(self, img):
        # write your codes here
        output = self.activation(self.c1(img))
        output = self.s2(output)
        output = self.activation(self.c3(output))
        output = self.s4(output)
        output = output.view(output.size(0), -1)
        output = self.activation(self.c5(output))
        output = self.activation(self.f6(output))
        output = self.output(output)
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()
        # write your codes here
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = nn.ReLU()
    def forward(self, img):
        # write your codes here
        output = self.activation(self.conv1(img))#32
        output = self.mp1(output)#16
        output = self.activation(self.conv2(output))#16
        output = self.mp2(output)#8
        output = self.activation(self.conv3(output))#8
        output = self.mp3(output)#4
        output = output.view(output.size(0), -1)
        output = self.activation(self.fc1(output))
        output = self.fc2(output)
        return output

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LeNet5 = LeNet5().to(device)
# summary(LeNet5, (1, 32,32))
# CustomMLP = CustomMLP().to(device)
# summary(CustomMLP, (1, 32,32))