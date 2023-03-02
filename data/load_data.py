import pickle

from data.generate_filename import generate_filepath
from data.preprocess_data import preprocess_data

from log import get_logger

log = get_logger(__name__)


def load_data(projection_type: str, is_permuted: bool):
    '''
    Loads the CICY dataset, including matrices (which are the input values of
    the problem) and Hodge numbers (which are the output values of the problem.

    :param projection_type: If 'dirichlet', then the dataset with Dirichlet projection
    pre-processing is returned. If 'combinatorial', then the dataset with
    combinatorial fundamental domain projection is returned. If any other string
    is passed, then the dataset without any fundamental domain pre-processing
    is returned.
    :param is_permuted: If True, then the dataset is loaded in which matrices
    had been randomly permuted before applying the fundamental domain projection.
    :param is_called_recursively: A technical parameter that is needed to allow
    the function to call itself if it creates the dataset that is to be loaded
    on the fly. If the function is called in another program, this should be set
    to False.

    :return: A list with two elements. The first is a nested list of shape
    (2, 12, 15) containing the CICY matrices. The second is a nested list of shape
    (2, 2) containing the Hodge numbers.
    '''
    file_name = generate_filepath(projection_type, is_permuted)

    while True:
        try:
            print(file_name)
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            log.warning(f'File {file_name} not found. Trying to automatically run preprocess_data.py to create the file.')
            preprocess_data(projection_type, is_permuted)
            log.info(f'Calling preprocess_data successful. The file should have been generated.')


if __name__ == '__main__':
    import numpy as np
    loaded_data = load_data('', False)
    print(np.array(loaded_data[0]).shape) # (7890, 12, 15)
    print(np.array(loaded_data[1]).shape) # (7890, 2)

    loaded_data = load_data('', True)
