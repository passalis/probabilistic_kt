import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def save_model(net, output_file='model.state'):
    """
    Saves a pytorch model
    :param net:
    :param output_file:
    :return:
    """
    torch.save(net.state_dict(), output_file)


def load_model(net, input_file='model.state'):
    """
    Loads a pytorch model
    :param net:
    :param input_file:
    :return:
    """
    state_dict = torch.load(input_file)
    net.load_state_dict(state_dict)


def train_model(net, optimizer, criterion, train_loader, epochs=10):
    """
    Trains a pytorch model
    :param net:
    :param optimizer:
    :param criterion:
    :param train_loader:
    :param epochs:
    :return:
    """
    for epoch in range(epochs):
        net.train()

        train_loss, correct, total = 0, 0, 0
        for (inputs, targets) in tqdm(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate statistics
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print("\nLoss, acc = ", train_loss, correct / total)


def get_labels(test_loader):
    """
    Extracts the labels from a loader
    :return:
    """
    labels = []
    for (inputs, targets) in tqdm(test_loader):
        labels.append(targets.numpy())

    return np.concatenate(labels).reshape((-1,))


def extract_features(net, test_loader):
    """
    Extracts features from a neural network
    :param net: a network that must implement net.get_features()
    :param test_loader:
    :return:
    """
    net.eval()

    features = []
    for (inputs, targets) in tqdm(test_loader):
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)

        outputs = net.get_features(inputs)
        outputs = outputs.cpu()
        features.append(outputs.data.numpy())

    return np.concatenate(features)

def get_raw_features(test_loader):
    """
    Extracts the raw input features
    :return:
    """
    features = []
    for (inputs, targets) in tqdm(test_loader):
        features.append(np.float16(inputs.numpy()))
    return np.concatenate(features)
