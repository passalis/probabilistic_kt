import pickle
import numpy as np


def print_results_line(model_name='-', pickle_path=None):
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)

    line = model_name + ' '

    m, t1, t2 = [], [] , []
    for i in range(len(results)):
        res = results[i]
        m.append(res['map'])
        t1.append(res['raw_precision'][49])
        t2.append(res['raw_precision'][199])

    line += '& ${%3.2f} \\pm {%3.2f}$ ' % (100 * np.mean(m), 100 * np.std(m))
    line += '& ${%3.2f} \\pm {%3.2f}$ ' % (100 * np.mean(t1), 100 * np.std(t1))
    line += '& ${%3.2f} \\pm {%3.2f}$ ' % (100 * np.mean(t2), 100 * np.std(t2))
    print(line, '\\\\')


def print_table():
    print_results_line(model_name='LBP', pickle_path='results/baseline_lbp.pickle')
    print("\\hline")
    print("\\hline")
    print_results_line(model_name='Hint (HoG)', pickle_path='results/hint.pickle')
    print_results_line(model_name='PKT (HoG)', pickle_path='results/kt.pickle')
    print_results_line(model_name='PKT (larger lr) (HoG)', pickle_path='results/kt_optimal.pickle')
    print_results_line(model_name='S-PKT (HoG)', pickle_path='results/kt_supervised.pickle')


if __name__ == '__main__':
    print_table()
