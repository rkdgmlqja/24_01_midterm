import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):

        # write your codes here
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    def __len__(self):

        # write your codes here
        return len(self.filenames)
    def __getitem__(self, idx):

        # write your codes here
        img_name = os.path.join(self.data_dir, self.filenames[idx])
        img = Image.open(img_name).convert('L')  # convert image to grayscale
        label = int(self.filenames[idx].split('_')[1].split('.')[0])  # extracts label from filename
        img = self.transform(img)  # Apply the preprocessing transformations
        return img, label

class MNIST_norm(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):

        # write your codes here
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.png')]

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation((-10,10)),
            transforms.RandomPerspective(0.1, 0.1)
        ])
    def __len__(self):

        # write your codes here
        return len(self.filenames)
    def __getitem__(self, idx):

        # write your codes here
        img_name = os.path.join(self.data_dir, self.filenames[idx])
        img = Image.open(img_name).convert('L')  # convert image to grayscale
        label = int(self.filenames[idx].split('_')[1].split('.')[0])  # extracts label from filename
        img = self.transform(img)  # Apply the preprocessing transformations
        return img, label

if __name__ == '__main__':
    dataset = MNIST(data_dir='path_to_your_data')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    for images, labels in dataloader:
        print(f"Batch size: {images.size(0)}")
        print(f"Image tensor size: {images.size()}")
        print(f"Label size: {labels.size()}")
