from exp_yf.yt_dataset import get_yt_loaders
from nn.retrieval_evaluation import retrieval_evaluation
import pickle

def evaluate_baseline(n_repeats=5):

    results = []
    for i in range(n_repeats):
        train_loader, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='lbp', seed=i)
        cur_res = retrieval_evaluation(None, database_loader, test_loader, raw=True)
        results.append(cur_res)
        print(cur_res)

    with open("results/baseline_lbp.pickle", 'wb') as f:
        pickle._dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    print("Running baseline evaluation...")
    evaluate_baseline()