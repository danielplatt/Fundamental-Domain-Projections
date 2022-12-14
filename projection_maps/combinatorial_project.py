import numpy as np
from projection_maps.auxiliary_functions.matrix_permutation_functions import argmax_nonflat, transposition, permuteMatrix
from functools import cmp_to_key #For user defined sorting

def row_ordering(row1: np.array, row2: np.array, averaging: bool)-> int:
    '''Determine which row is larger, either lexicographically or by their average.
    :param row1, row2: rows that are compared
    :averaging: if True, then rows are compared based on their average. If False, then lexicographically
    :return: 1 if x2>x1, -1 if x1<x2, 0 if x1==x2
    '''
    x1 = row1[0]
    x2 = row2[0]
    if averaging == True:
        x1 = np.sum(row1)
        x2 = np.sum(row2)
    if x1 > x2:
        return -1
    if x1 == x2:
        return 0
    if x1 < x2:
        return 1

def max_to_topleft(matrix: np.array)->np.array:
    '''Apply row and column transpositions to move extrema to top left.
    :param matrix: matrix to be permuted
    :return: matrix with max permuted to top left corner
    '''
    k, m = matrix.shape
    i, j = argmax_nonflat(matrix)
    return np.dot(np.dot(transposition(i, 0, k), matrix), transposition(0, j, m))


def sort_rows(matrix: np.array, averaging: bool)-> np.array:
    '''
    Apply row permutations to a matrix, with the effect of ordering the rows from largest to smallest.
    :param matrix: matrix to be permuted
    :return: sorted matrix
    '''
    return sorted(matrix, key=cmp_to_key(lambda x, y: row_ordering(x, y,averaging)))


def sort_cols(matrix, averaging):
    '''
    Apply column permutations to a matrix, with the effect of ordering the column from largest to smallest.
    :param matrix: matrix to be permuted
    :return: sorted matrix
    '''
    return np.transpose(sort_rows(np.transpose(matrix), averaging))

def combinatorial_project(matrix: np.array, ascending=True, averaging='True', pert_is_permuted=False) -> np.array:
    '''Apply combinatorial projection map to a matrix of arbitrary dimensions. The projection proceeds in four steps
     1) perturb, 2) move min entry to top left, 3) sort by first row and column, 4) unperturb.
     :param matrix: matrix to be permuted
     :param ascending: if True, minimum element will be in the top left corner and rows and columns are ordered from smallest largest. If False, maximum is in the top left corner and rows and columns are order from largest to smallest
     :param averaging: If True, sorting of rows and columns is based on average values. If False, it is lexicographical
     :param pert_is_permuted: If False, pertubation matrix is identical for every execution. If True, the pert matrix is randomly permuted
     :return: permuted matrix
    '''
    if ascending:
        return -combinatorial_project(matrix=-matrix, ascending=False, averaging=averaging,    pert_is_permuted=pert_is_permuted)


    def combinatorial_unperturbed(matrix):
        if averaging==False:
            matrix = max_to_topleft(matrix)
        matrix = sort_cols(matrix, averaging)
        matrix = sort_rows(matrix, averaging)
        return matrix

    (m, k) = matrix.shape
    P = np.linspace(-0.4, 0.4, m * k).reshape((m, k))
    if pert_is_permuted:
        P=permuteMatrix(P)
    return np.round(combinatorial_unperturbed(matrix + P))


if __name__ == '__main__':
    testlist=np.random.randint(6,size=(8,4,4))
    for A in testlist:
        A_proj = combinatorial_project(matrix=A,ascending = False ,averaging=False)
        print('original matrix \n '+str(A)+'\n comb_ord \n '+str(A_proj)+ '\n')


