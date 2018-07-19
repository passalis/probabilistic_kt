import torch.nn as nn
import torch.optim as optim
from nn.retrieval_evaluation import evaluate_model_retrieval
from exp_cifar.cifar_dataset import cifar10_loader
from models.cifar_tiny import Cifar_Tiny
from models.resnet import ResNet18
from nn.nn_utils import train_model, save_model


def train_cifar10_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50], output_path='resnet18_cifar10.model'):
    """
    Trains a baseline (classification model)
    :param net: the network to be trained
    :param learning_rates: the learning rates to be used during the training
    :param iters: number of epochs using each of the supplied learning rates
    :param output_path: path to save the trained model
    :return:
    """

    # Load data
    train_loader, test_loader, _ = cifar10_loader(batch_size=128)

    # Define loss
    criterion = nn.CrossEntropyLoss()

    for lr, iter in zip(learning_rates, iters):
        print("Training with lr=%f for %d iters" % (lr, iter))
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train_model(net, optimizer, criterion, train_loader, epochs=iter)
        save_model(net, output_file=output_path)


def train_cifar_models():
    """
    Trains the baselines teacher/students
    :return:
    """

    # ResNet training
    net = ResNet18()
    net.cuda()
    train_cifar10_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50],
                        output_path='models/resnet18_cifar10.model')

    # Cifar Tiny
    net = Cifar_Tiny()
    net.cuda()
    train_cifar10_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50],
                        output_path='models/tiny_cifar10.model')

def evaluate_cifar_models_retrieval():
    """
    Evaluates the baselines teacher/students
    :return:
    """
    evaluate_model_retrieval(net=Cifar_Tiny(num_classes=10), path='models/tiny_cifar10.model',
                             result_path='results/tiny_cifar10_baseline.pickle')

    evaluate_model_retrieval(net=ResNet18(num_classes=10), path='models/resnet18_cifar10.model',
                   result_path='results/resnet18_cifar10_baseline.pickle')



if __name__ == '__main__':
    # Training the teacher model takes approximately a day, so you can use the pretrained model
    # train_cifar_models()

    evaluate_cifar_models_retrieval()


