import torch
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


def distillation_loss(output, target, T):
    """
    Distillation Loss
    :param output:
    :param target:
    :param T:
    :return:
    """
    output = F.log_softmax(output / T)
    target = F.softmax(target / T)
    loss = -torch.sum(target * output) / output.size()[0]
    return loss


def unsupervised_distillation(net, net_to_distill, transfer_loader, epochs=1, lr=0.0001, T=2):
    """
    Performs unsupervised neural network distillation
    :param net:
    :param net_to_distill:
    :param transfer_loader:
    :param epochs:
    :param lr:
    :return:
    """

    optimizer = optim.Adam(params=net.parameters(), lr=lr)

    for epoch in range(epochs):

        net.train()
        net_to_distill.eval()

        train_loss = 0

        for (inputs, targets) in tqdm(transfer_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # Feed forward the network and update
            optimizer.zero_grad()

            # Get the data
            inputs = inputs.cuda()
            output_target = net_to_distill(Variable(inputs))
            outputs = net(Variable(inputs))

            # Get the loss
            loss = distillation_loss(outputs, output_target, T=T)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data.item()

        print("\nLoss  = ", train_loss)
