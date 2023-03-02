import numpy as np
import pickle
from tqdm import tqdm

from data.generate_filename import generate_filepath
from data.create_cicy_original_pckl import create_cicy_original_pckl
from projection_maps.combinatorial_project import combinatorial_project
from projection_maps.dirichlet_project import dirichlet_project
from projection_maps.auxiliary_functions.matrix_auxiliary_functions import permuteSingleMatrix

from log import get_logger

log = get_logger(__name__)


def preprocess_data(projection_type: str, is_permuted: bool) -> np.array:
    '''Requires file cicy_original.pckl, creates files
    cicy_permuted.pckl,
    cicy_dirichlet_from_original.pckl,
    cicy_dirichlet_from_permuted.pckl,
    cicy_combinatorial_from_original.pckl,
    cicy_combinatorial_from_permuted.pckl
    '''

    file_name = generate_filepath(None, False)
    while True:
        try:
            with open(file_name, 'rb') as f:
                original = pickle.load(f)
                hodge_numbers = original[1]
                matrices = original[0]
                break
        except FileNotFoundError as e:
            log.warning(f'File {file_name} not found. Trying to run create_cicy_original_pckl.py to generate it now...')
            create_cicy_original_pckl()
            log.info(f'File {file_name} should have been created now. Trying to continue running preprocess_data.py...')

    if is_permuted:
        matrices = [permuteSingleMatrix(matrix) for matrix in matrices]

    if projection_type == 'dirichlet':
        new_matrices = []
        for matrix in tqdm(matrices):
            new_matrices += [dirichlet_project(matrix)]
        matrices = new_matrices
    elif projection_type == 'combinatorial':
        new_matrices = []
        for matrix in tqdm(matrices):
            new_matrices += [dirichlet_project(matrix)]
        matrices = new_matrices

    file_name = generate_filepath(projection_type, is_permuted)
    with open(file_name, 'wb') as f:
        pickle.dump([matrices, hodge_numbers], f)


if __name__ == '__main__':
    preprocess_data(None, True)
    preprocess_data('dirichlet', True)
    preprocess_data('combinatorial', True)
    preprocess_data('dirichlet', False)
    preprocess_data('combinatorial', False)
