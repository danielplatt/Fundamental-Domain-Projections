import numpy as np
from numpy.linalg import matrix_power

def transpose_matrix_entryfunction(k,m,index):
    i,j=index
    if (i==k and j==m) or (i==m and j==k):
        return 1
    if i==j and not (i==k or i==m):
        return 1
    return 0

def cycle_matrix_entryfunction(n,index):
    i,j=index
    if j==i+1:
        return 1
    if i==n-1 and j==0:
        return 1
    return 0

def transposition(k,m,n):
    return np.array([[transpose_matrix_entryfunction(k,m,[i,j]) for i in range(n)] for j in range(n)])

def cycle(power,n):
    #n=dimension of the square matrix, this is to produce cyclic permutations of a sq matrix
    A=np.array([[cycle_matrix_entryfunction(n, [i, j]) for i in range(n)] for j in range(n)])
    return matrix_power(A,power)

def flatten_onelevel(arr):
    sh=list(np.shape(arr))
    newshape=sh[2:]
    newshape.insert(0,sh[0]*sh[1])
    return np.reshape(arr,tuple(newshape))

def matrix_equality(matrix1,matrix2):
    if np.array_equal(matrix1,matrix2):
        return 1
    else:
        return 0
def permuteMatrix(matrix):
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

#this Function returns the position of the maximum in an array, np.argmax doesn't give a tupel but just a number
def argmax_nonflat(matrix):
    return np.unravel_index(matrix.argmax(), matrix.shape)

#Taking the inner product of two matrices, treating both as a vector
def matrix_innerproduct(A,B):
    assert np.shape(A)==np.shape(B)
    k,m=np.shape(A)
    prod=k*m
    return np.dot(A.reshape(prod),B.reshape(prod))

def matrix_order(x,y,x0):
    if x0=='Daniel':
        a=x.flatten()
        b=y.flatten()
        if (a==b).all():
            return 0
        idx = np.where( (a>b) != (a<b) )[0][0]
        if a[idx] < b[idx]:
            return -1
        else:
            return 1
    else:
        a=matrix_innerproduct(x,x0)
        b=matrix_innerproduct(y, x0)
        if a<b:
            return -1
        if a==b:
            return 0
        if b<a:
            return 1