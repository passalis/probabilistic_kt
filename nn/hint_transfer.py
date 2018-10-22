import torch
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm



def unsupervised_hint_transfer(net, net_to_distill, transfer_loader, epochs=1, lr=0.0001, W=None):
    """
    Performs unsupervised neural network hint-based transfer
    :param net:
    :param net_to_distill:
    :param transfer_loader:
    :param epochs:
    :param lr:
    :return:
    """

    if W is None:
        # Get the shapes of the layers
        net_to_distill.eval()
        net.eval()
        (inputs, targets) = next(iter(transfer_loader))
        inputs, targets = inputs.cuda(), targets.cuda()
        output_target = net_to_distill.get_features(Variable(inputs))
        output_net = net.get_features(Variable(inputs))
        in_feats, out_feats = output_target.size()[1], output_net.size()[1]

        W = torch.nn.Linear(in_feats, out_feats, bias=False)
        W.cuda()

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
            output_target = net_to_distill.get_features(Variable(inputs))
            output_target_reduced = W(output_target)
            output_net = net.get_features(Variable(inputs))

            loss = torch.sum((output_net - output_target_reduced) ** 2) / output_net.size()[0]

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data.item()

        print("\nLoss  = ", train_loss)

    return W



def unsupervised_hint_transfer_handcrafted(net, transfer_loader, epochs=1, lr=0.0001, W=None):
    """
    Performs unsupervised neural network hint-based transfer
    :param net:
    :param net_to_distill:
    :param transfer_loader:
    :param epochs:
    :param lr:
    :return:
    """

    if W is None:
        # Get the shapes of the layers
        net.eval()
        (inputs, features, targets) = next(iter(transfer_loader))
        inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
        output_target = Variable(features)
        output_net = net.get_features(Variable(inputs))
        out_feats, in_feats = output_target.size()[1], output_net.size()[1]
        W = torch.nn.Linear(out_feats, in_feats, bias=False)
        W.cuda()

    optimizer = optim.Adam(params=net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()

        train_loss = 0

        for (inputs, features, targets) in tqdm(transfer_loader):
            inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
            output_target = Variable(features)

            optimizer.zero_grad()
            output_target_reduced = W(output_target)
            output_net = net.get_features(Variable(inputs))

            loss = torch.sum((output_net - output_target_reduced) ** 2) / output_net.size()[0]

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data.item()

        print("\nLoss  = ", train_loss)

    return W


def unsupervised_hint_transfer_optimized(net, net_to_distill, transfer_loader, epochs=1, lr=0.0001, W=None):
    """
    Performs unsupervised neural network hint-based transfer (the projection is jointly optimized)
    :param net:
    :param net_to_distill:
    :param transfer_loader:
    :param epochs:
    :param lr:
    :return:
    """

    if W is None:
        # Get the shapes of the layers
        net_to_distill.eval()
        net.eval()
        (inputs, targets) = next(iter(transfer_loader))
        inputs, targets = inputs.cuda(), targets.cuda()
        output_target = net_to_distill.get_features(Variable(inputs))
        output_net = net.get_features(Variable(inputs))
        out_feats, in_feats = output_target.size()[1], output_net.size()[1]

        W = torch.nn.Linear(in_feats, out_feats, bias=False)
        W.cuda()

    params = list(net.parameters())
    params.extend(list(W.parameters()))
    optimizer = optim.Adam(params=params, lr=lr)

    for epoch in range(epochs):
        net.train()
        net_to_distill.eval()

        train_loss = 0

        for (inputs, targets) in tqdm(transfer_loader):
            # Feed forward the network and update
            optimizer.zero_grad()

            inputs, targets = inputs.cuda(), targets.cuda()

            # Get the data
            inputs = inputs.cuda()
            output_target = net_to_distill.get_features(Variable(inputs))
            output_net = net.get_features(Variable(inputs))
            output_net_expanded = W(output_net)
            loss = torch.sum((output_target - output_net_expanded) ** 2) / output_net.size()[0]

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data.item()

        print("\nLoss  = ", train_loss)

    return W
