import dataset
from model import LeNet5, CustomMLP
# import some packages you need here
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in trn_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    trn_loss = running_loss / len(trn_loader)
    acc = 100. * correct / total
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tst_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    tst_loss = running_loss / len(tst_loader)
    acc = 100. * correct / total
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    batch_size = 1024
    learning_rate = 0.01
    momentum = 0.9
    epochs = 10


    #train_dataset = dataset.MNIST("/home/sungrae/khb/24_1/data/train")
    train_dataset = dataset.MNIST_norm("/home/sungrae/khb/24_1/data/train")
    test_dataset = dataset.MNIST("/home/sungrae/khb/24_1/data/test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    LeNet = LeNet5().to(device)
    MLP = CustomMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(LeNet.parameters(), lr=learning_rate, momentum=momentum)

    criterion1 = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(MLP.parameters(), lr=learning_rate, momentum=momentum)


    LeNet_train_losses, LeNet_test_losses, LeNet_train_accuracies, LeNet_test_accuracies = [], [], [], []
    MLP_train_losses, MLP_test_losses, MLP_train_accuracies, MLP_test_accuracies = [], [], [], []

    for epoch in range(epochs):
        MLP_trn_loss, MLP_trn_acc = train(MLP, train_loader, device, criterion1, optimizer1)
        MLP_tst_loss, MLP_tst_acc = test(MLP, test_loader, device, criterion1)
        MLP_train_losses.append(MLP_trn_loss)
        MLP_train_accuracies.append(MLP_trn_acc)
        MLP_test_losses.append(MLP_tst_loss)
        MLP_test_accuracies.append(MLP_tst_acc)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'MLP Train Loss: {MLP_trn_loss:.4f}, MLP Train Accuracy: {MLP_trn_acc:.2f}%')
        print(f'MLP Test Loss: {MLP_tst_loss:.4f}, MLP Test Accuracy: {MLP_tst_acc:.2f}%')

    for epoch in range(epochs):
        LeNet_trn_loss, LeNet_trn_acc = train(LeNet, train_loader, device, criterion, optimizer)
        LeNet_tst_loss, LeNet_tst_acc = test(LeNet, test_loader, device, criterion)
        LeNet_train_losses.append(LeNet_trn_loss)
        LeNet_train_accuracies.append(LeNet_trn_acc)
        LeNet_test_losses.append(LeNet_tst_loss)
        LeNet_test_accuracies.append(LeNet_tst_acc)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'LeNet Train Loss: {LeNet_trn_loss:.4f}, LeNet Train Accuracy: {LeNet_trn_acc:.2f}%')
        print(f'LeNet Test Loss: {LeNet_tst_loss:.4f}, LeNet Test Accuracy: {LeNet_tst_acc:.2f}%')

    fig, axs = plt.subplots(4, 2, figsize=(15, 20))  # Increase the subplot to 4x2
    fig.suptitle('Training and Testing Metrics for LeNet and MLP')

    # LeNet Plots
    axs[0, 0].plot(range(1, epochs + 1), LeNet_train_losses, 'g-')
    axs[0, 0].set_title('LeNet Training Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')

    axs[0, 1].plot(range(1, epochs + 1), LeNet_test_losses, 'b--')
    axs[0, 1].set_title('LeNet Testing Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')

    axs[1, 0].plot(range(1, epochs + 1), LeNet_train_accuracies, 'g-')
    axs[1, 0].set_title('LeNet Training Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy (%)')

    axs[1, 1].plot(range(1, epochs + 1), LeNet_test_accuracies, 'b--')
    axs[1, 1].set_title('LeNet Testing Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Accuracy (%)')

    # MLP Plots
    axs[2, 0].plot(range(1, epochs + 1), MLP_train_losses, 'r-')
    axs[2, 0].set_title('MLP Training Loss')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Loss')

    axs[2, 1].plot(range(1, epochs + 1), MLP_test_losses, 'm--')
    axs[2, 1].set_title('MLP Testing Loss')
    axs[2, 1].set_xlabel('Epochs')
    axs[2, 1].set_ylabel('Loss')

    axs[3, 0].plot(range(1, epochs + 1), MLP_train_accuracies, 'r-')
    axs[3, 0].set_title('MLP Training Accuracy')
    axs[3, 0].set_xlabel('Epochs')
    axs[3, 0].set_ylabel('Accuracy (%)')

    axs[3, 1].plot(range(1, epochs + 1), MLP_test_accuracies, 'm--')
    axs[3, 1].set_title('MLP Testing Accuracy')
    axs[3, 1].set_xlabel('Epochs')
    axs[3, 1].set_ylabel('Accuracy (%)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust the layout to make room for all subplots
    plt.show()


if __name__ == '__main__':
    main()
