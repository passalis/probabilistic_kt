from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from time import time
from nn.nn_utils import get_labels, extract_features, get_raw_features
import numpy as np
import pickle
from exp_cifar.cifar_dataset import cifar10_loader
from nn.nn_utils import load_model


class Database(object):
    def __init__(self, database_vectors, targets, metric='cosine'):
        self.nn = NearestNeighbors(n_neighbors=database_vectors.shape[0], algorithm='brute', metric=metric)
        self.nn.fit(database_vectors)
        self.targets = np.cast[np.int](targets)
        bins = np.bincount(self.targets)
        idx = np.nonzero(bins)[0]
        self.instances_per_target = dict(zip(idx, bins[idx]))
        self.number_of_instances = float(len(targets))
        self.recall_levels = np.arange(0, 1.01, 0.1)
        self.fine_recall_levels = np.arange(0, 1.01, 0.05)

    def get_binary_relevances(self, queries, targets):
        """
        Executes the queries and returns the binary relevance vectors (one vector for each query)
        :param queries: the queries
        :param targets: the label of each query
        :return:
        """
        distances, indices = self.nn.kneighbors(queries)
        relevant_vectors = np.zeros_like(indices)
        for i in range(targets.shape[0]):
            relevant_vectors[i, :] = self.targets[indices[i, :]] == targets[i]
        return relevant_vectors

    def get_metrics(self, relevant_vectors, targets):
        """
        Evaluates the retrieval performance
        :param relevant_vectors: the relevant vectors for each query
        :param targets: labels of the queries
        :return:
        """
        # Calculate precisions per query
        precision = np.cumsum(relevant_vectors, axis=1) / np.arange(1, self.number_of_instances + 1)

        # Calculate interpolated precision
        for i in reversed(range(len(precision) - 1)):
            precision[:, i] = np.maximum(precision[:, i], precision[:, i + 1])

        # Calculate recall per query
        instances_per_query = np.zeros((targets.shape[0], 1))
        for i in range(targets.shape[0]):
            instances_per_query[i] = self.instances_per_target[targets[i]]
        recall = np.cumsum(relevant_vectors, axis=1) / instances_per_query

        # Calculate precision @ 11 recall point
        precision_at_recall_levels = np.zeros((targets.shape[0], self.recall_levels.shape[0]))
        for i in range(len(self.recall_levels)):
            idx = np.argmin(np.abs(recall - self.recall_levels[i]), axis=1)
            precision_at_recall_levels[:, i] = precision[np.arange(targets.shape[0]), idx]

        # Calculate fine-grained precision
        precision_at_fine_recall_levels = np.zeros((targets.shape[0], self.fine_recall_levels.shape[0]))
        for i in range(len(self.fine_recall_levels)):
            idx = np.argmin(np.abs(recall - self.fine_recall_levels[i]), axis=1)
            precision_at_fine_recall_levels[:, i] = precision[np.arange(targets.shape[0]), idx]

        # Calculate the means values of the metrics
        ap = np.mean(precision_at_recall_levels, axis=1)
        m_ap = np.mean(ap)
        interpolated_precision = np.mean(precision, axis=0)
        interpolated_fine_precision = np.mean(precision_at_fine_recall_levels, axis=0)

        return m_ap, interpolated_precision, interpolated_fine_precision, self.fine_recall_levels,

    def evaluate(self, queries, targets, batch_size=128):
        """
        Evaluates the performance of the database using the following metrics: interpolated map, interpolated precision,
        and precision-recall curve
        :param queries: the queries
        :param targets: the labels
        :return: the evaluated metrics
        """
        n_batches = len(targets) // batch_size
        m_ap, fine_precision, raw_precision = None, None, None

        for i in tqdm(range(n_batches)):
            cur_queries = queries[i * batch_size:(i + 1) * batch_size]
            cur_targets = targets[i * batch_size:(i + 1) * batch_size]

            relevant_vectors = self.get_binary_relevances(cur_queries, cur_targets)
            (c_m_ap, c_raw_precision, c_fine_precision, self.fine_recall_levels,) = \
                self.get_metrics(relevant_vectors, cur_targets)

            if m_ap is None:
                m_ap = c_m_ap * batch_size
                fine_precision = c_fine_precision * batch_size
                raw_precision = c_raw_precision * batch_size
            else:
                m_ap += c_m_ap * batch_size
                fine_precision += c_fine_precision * batch_size
                raw_precision += c_raw_precision * batch_size

        if batch_size * n_batches < len(targets):
            cur_queries = queries[batch_size * n_batches:]
            cur_targets = targets[batch_size * n_batches:]

            relevant_vectors = self.get_binary_relevances(cur_queries, cur_targets)
            (c_m_ap, c_raw_precision, c_fine_precision, self.fine_recall_levels,) = \
                self.get_metrics(relevant_vectors, cur_targets)

            m_ap += c_m_ap * len(cur_targets)
            fine_precision += c_fine_precision * len(cur_targets)
            raw_precision += c_raw_precision * len(cur_targets)

        m_ap = m_ap / float(len(targets))
        fine_precision = fine_precision / float(len(targets))
        raw_precision = raw_precision / float(len(targets))

        results = {'map': m_ap, 'precision': fine_precision, 'recall_levels': self.fine_recall_levels,
                   'raw_precision': raw_precision}

        return results


def retrieval_evaluation(net, train_loader, test_loader, metric='cosine', raw=False):
    """
    Evalutes a pytorch model using a retrieval setup
    :param net:
    :param train_loader:
    :param test_loader:
    :param metric:
    :return:
    """

    # Get the labels
    train_labels = get_labels(train_loader)
    test_labels = get_labels(test_loader)

    # Get the features
    a = time()
    if raw:
        train_features = get_raw_features(train_loader)
        test_features = get_raw_features(test_loader)
        ff_time = 0
    else:
        train_features = extract_features(net, train_loader)
        ff_time = (time() - a) / float(len(train_labels))
        test_features = extract_features(net, test_loader)

    # Evaluate the model
    database = Database(train_features, train_labels, metric=metric)
    a = time()
    results = database.evaluate(test_features, test_labels, batch_size=128)
    retrieval_time = (time() - a) / float(len(test_labels))

    results['retrieval_time'] = retrieval_time
    results['ff_time'] = ff_time

    return results


def evaluate_model_retrieval(path='', net=None, result_path='', dataset_name='cifar10', dataset_loader=cifar10_loader):
    """
    Wrapper function for the evaluation that also saves the results into the appropriate output files
    :param path:
    :param net:
    :param result_path:
    :param dataset_name:
    :param dataset_loader:
    :return:
    """
    # If a path is supplied load the model
    if path != '':
        net.cuda()
        load_model(net, path)

    _, test_loader, train_loader = dataset_loader(batch_size=128)
    results = retrieval_evaluation(net, train_loader, test_loader)

    results = {dataset_name: results}
    with open(result_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
