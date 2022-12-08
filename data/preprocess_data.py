import numpy as np
import pickle
from tqdm import tqdm

from data.generate_filename import generate_filename
from fundamental_domain_projections.combinatorial_project import combinatorial_project
from fundamental_domain_projections.dirichlet_project import dirichlet_project


def permuteSingleMatrix(matrix: np.array) -> np.array:
    '''
    :param: matrix: A two-dimensional numpy array, e.g. np.array([[1,2,3],[4,5,6]])
    :return: Rows and columns of this matrix randomly permuted, eg. np.array([[5,4,6],[2,1,3]])
    '''
    numberOfRows = len(matrix)
    numberofColumns = len(matrix[0])
    rowPermutation = list(np.random.permutation(numberOfRows))
    columnPermutation = list(np.random.permutation(numberofColumns))
    newMatrix = matrix[:,columnPermutation]
    newMatrix = newMatrix[rowPermutation]
    return newMatrix

def preprocess_data(projection_type: str, is_permuted: bool) -> np.array:
    '''Requires file cicy_original.pckl, creates files
    cicy_permuted.pckl,
    cicy_dirichlet_from_original.pckl,
    cicy_dirichlet_from_permuted.pckl,
    cicy_combinatorial_from_original.pckl,
    cicy_combinatorial_from_permuted.pckl
    '''
    with open('cicy_original.pckl', 'rb') as f:
        original = pickle.load(f)
        hodge_numbers = original[1]
        matrices = original[0]

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

    file_name = generate_filename(projection_type, is_permuted)
    with open(file_name, 'wb') as f:
        pickle.dump([matrices, hodge_numbers], f)


if __name__ == '__main__':
    preprocess_data(None, True)
    preprocess_data('dirichlet', True)
    preprocess_data('combinatorial', True)
    preprocess_data('dirichlet', False)
    preprocess_data('combinatorial', False)
