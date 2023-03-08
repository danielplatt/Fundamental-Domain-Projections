from sklearn.model_selection import train_test_split
import numpy as np

from models.BullHeJejjalaMishraCNN import train_bull_he_jejjala_mishra_network
from models.ErbinFinotelloInception import train_erbin_finotello
from models.Hartford import train_hartford_network
from models.HeNN import train_he_network

from data.load_data import load_data


def training(trainingtask, datasetpreprocessing='', datasetpermuted=False):
    data = load_data(datasetpreprocessing, datasetpermuted)
    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]), np.array(data[1])[:, 0], test_size=0.5)
    print(f'Beginning training task {trainingtask}')
    print(f'Train on dataset with preprocessing "{datasetpreprocessing}"')
    print(f'Has dataset been permuted before pre-processing? {datasetpermuted}')
    accuracy_result = trainingtask(X_train, y_train, X_test, y_test)
    return accuracy_result

def main():
    all_accuracies = []
    for trainingtask in [train_bull_he_jejjala_mishra_network, train_hartford_network, train_he_network]:
        for datasetpermuted in [False, True]:
            this_acc = training(trainingtask, datasetpreprocessing='', datasetpermuted=datasetpermuted)
            all_accuracies += [(trainingtask.__name__, '', datasetpermuted, this_acc)]
            print(all_accuracies)

    for datasetpreprocessing in ['', 'dirichlet', 'combinatorial']:
        for datasetpermuted in [False, True]:
            this_acc = training(train_erbin_finotello, datasetpreprocessing=datasetpreprocessing, datasetpermuted=datasetpermuted)
            all_accuracies += [('train_erbin_finotello', datasetpreprocessing, datasetpermuted, this_acc)]
            print(all_accuracies)

    return all_accuracies


if __name__ == '__main__':
    print(main())
