import torch
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import numpy as np


def knowledge_transfer(net, net_to_distill, transfer_loader, epochs=1, lr=0.0001, supervised_weight=0):
    """
    Performs unsupervised neural network knowledge transfer
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
        counter = 1
        for (inputs, targets) in tqdm(transfer_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # Feed forward the network and update
            optimizer.zero_grad()

            # # Get the data
            output_target = net_to_distill.get_features(Variable(inputs))
            outputs_net = net.get_features(Variable(inputs))

            # Get the loss
            if supervised_weight > 0:
                loss = cosine_similarity_loss(outputs_net, output_target) + \
                       supervised_weight * supervised_loss(outputs_net, targets)
            else:
                loss = cosine_similarity_loss(outputs_net, output_target)

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data.item()
            counter += 1

        train_loss = train_loss / float(counter)
        print("\n Epoch = ", epoch, " Loss  = ", train_loss)


def knowledge_transfer_handcrafted(net, transfer_loader, epochs=1, lr=0.0001, supervised_weight=0):
    optimizer = optim.Adam(params=net.parameters(), lr=lr)

    for epoch in range(epochs):

        net.train()

        train_loss = 0
        counter = 1
        for (inputs, features, targets) in tqdm(transfer_loader):
            inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
            output_target = Variable(features)

            # Feed forward the network and update
            optimizer.zero_grad()

            # # Get the data
            outputs_net = net.get_features(Variable(inputs))

            # Get the loss
            loss = cosine_similarity_loss(outputs_net, output_target) + \
                   supervised_weight * supervised_loss(outputs_net, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data[0]
            counter += 1

        train_loss = train_loss / float(counter)
        print("\n Epoch = ", epoch, " Loss  = ", train_loss)


def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss


def supervised_loss(output_net, targets, eps=0.0000001):
    labels = targets.cpu().numpy()
    target_sim = np.zeros((labels.shape[0], labels.shape[0]), dtype='float32')
    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            if labels[i] == labels[j]:
                target_sim[i, j] = 1.0
            else:
                target_sim[i, j] = 0

    target_similarity = torch.from_numpy(target_sim).cuda()
    target_similarity = Variable(target_similarity)

    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss
