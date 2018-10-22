import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

import pickle
from nn.nn_utils import save_model
from exp_yf.yt_dataset import get_yt_loaders
from models.yt import YT_Small
from nn.pkt import knowledge_transfer_handcrafted
from nn.hint_transfer import unsupervised_hint_transfer_handcrafted
from nn.retrieval_evaluation import retrieval_evaluation


def perform_kt_transfer(kt_type='hint', epochs=10):
    results = []
    for i in range(5):

        train_loader, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='transfer', seed=i)
        net = YT_Small()
        net.cuda()

        if kt_type == 'hint':
            unsupervised_hint_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.0001)
        elif kt_type == 'kt':
            knowledge_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.0001)
        elif kt_type == 'kt_optimal':
            knowledge_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.001)
        elif kt_type == 'kt_supervised':
            knowledge_transfer_handcrafted(net, train_loader, epochs=epochs, lr=0.0001, supervised_weight=0.001)
        save_model(net, 'models/' + kt_type + '_' + str(i) + '.model')

        train_loader, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='image', seed=i)
        cur_res = retrieval_evaluation(net, database_loader, test_loader)
        results.append(cur_res)
        print(cur_res)

    with open('results/' + kt_type + '.pickle', 'wb') as f:
        pickle._dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    perform_kt_transfer('hint')
    perform_kt_transfer('kt')
    perform_kt_transfer('kt_supervised')

    ## Additional experiments
    # KT is also stable when a larger learning rate is used
    perform_kt_transfer('kt_optimal')
