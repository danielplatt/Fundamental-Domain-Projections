import numpy as np
from numpy.linalg import matrix_power

def pmatrix_trans(k:int, m: int, n:int):
    '''Permutation matrix of size nxn respresenting a transposition of element k with element m'''
    def transpose_matrix_entryfunction(k, m, index):
        '''A function whose output will be the transposition k-m-matrix at a given index.'''
        i, j = index
        if (i == k and j == m) or (i == m and j == k):
            return 1
        if i == j and not (i == k or i == m):
            return 1
        return 0
    return np.array([[transpose_matrix_entryfunction(k,m,[i,j]) for i in range(n)] for j in range(n)])

def pmatrix_cycle(power: int, n: int) -> np.array:
    '''Permutation matrix representing a cycle to the power n. '''
    def cycle_matrix_entryfunction(n, index):
        i, j = index
        if j == i + 1:
            return 1
        if i == n - 1 and j == 0:
            return 1
        return 0
    #n=dimension of the square matrix, this is to produce cyclic permutations of a sq matrix
    A=np.array([[cycle_matrix_entryfunction(n, [i, j]) for i in range(n)] for j in range(n)])
    return matrix_power(A,power)


def permuteMatrix(matrix: np.array) -> np.array:
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


def matrix_innerproduct(matrix1: np.array,matrix2: np.array) -> float:
    '''Flatten two matrices and return their inner product.'''
    return np.dot(matrix1.reshape(-1),matrix2.reshape(-1))

def matrix_order(matrix1: np.array, matrix2: np.array, x0) -> int:
    '''Define an order relation on the space of matrices by comparing the inner product with a fixed matrix x0.
    :param matrix1, matrix2: matrices to be compared
    :param x0: matrix defining the order relation, can also be 'Daniel', which imposes lexicographical order
    :return: 1 if matrix1<matrix2, 1 if matrix2>matrix1, 0 if matrix1=matrix2
    '''
    if x0=='Daniel':
        a=matrix1.flatten()
        b=matrix2.flatten()
        if (a==b).all():
            return 0
        #find the first index where a and b are unequal
        idx = np.where( (a>b) != (a<b) )[0][0]
        if a[idx] < b[idx]:
            return -1
        else:
            return 1
    else:
        a=matrix_innerproduct(matrix1, x0)
        b=matrix_innerproduct(matrix2, x0)
        if a<b:
            return -1
        if a==b:
            return 0
        if b<a:
            return 1

def pad_matrix(matrix, pad_with=1):
    '''Embebds a matrix of smaller size into a 12x15 matrix by padding it with zeros and additional elements on the main diagonal
    :param matrix: matrix to be padded
    :param pad_with: float (or None)  that is put on the diagonal of the larger matrix
    '''
    if pad_with== None:
        return matrix
    else:
        c,r=np.shape(matrix)
        b=np.zeros((12,15))
        b[:c,:r]=matrix
        for i in range(c,12):
            # write pad_with on the diagonal
            b[i,r+i-c]=pad_with
        return b

if __name__=='__main__':
    arr = np.array([[1,2,3],[4,5,6]])
    print(matrix_innerproduct(arr,arr))
    #print(np.reshape(arr,(-1)))
    #print(flatten_onelevel(arr))
