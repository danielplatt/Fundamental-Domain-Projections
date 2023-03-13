import numpy as np
from projection_maps.auxiliary_functions.generating_sets import *
from projection_maps.auxiliary_functions.matrix_auxiliary_functions import *
from functools import cmp_to_key #For user defined sorting

def gradient_ascent(x: np.array,x0='Daniel',gen_type='neighbourtranspositions'):
    ''' Perform a discrete gradient ascent on the set of matrixes to maximise the inner product with a fixed matrix x0 along a finite but large group orbit. The allowed group actions are subgroups of the permutation group and either permute rows, columns or apply transpositions. A detailed description is found in Appendix F.3.
    :param x: start value for the algorithm, which optimises along the group orbit of x
    :param x0: fixed matrix, the algorithm maximises the inner product with x0. A special case is x0=='Daniel', which gives a faster way to perform lexicographical ordering
    :param gen_type: the type of generators for the group action
    :return: matrix maximising the inner product with x0 along the group orbit of x
    '''
    matrix_dim=np.shape(x)

## Get generators ##

# For matrix permutations there are three types of generators. In the CICY example, we don't use transpositions
    gen_row=generators(gen_type,matrix_dim,'row').elements
    gen_col=generators(gen_type,matrix_dim,'col').elements
    trans=generators(gen_type,matrix_dim,'diagonal').elements

    len_row=len(gen_row)
    len_col=len(gen_col)
    len_trans=len(trans)

## find maximum iteratively
    max_found=False
    while max_found==False:
        i=0
        while i<len_row+len_col+len_trans:
            #depending on i, we apply to x a row permutation, column permutation or transposition
            if i<len_row:
                y = np.dot(gen_row[i], x)
            if i>=len_row and i<len_row+len_col:
                y = np.dot(x, gen_col[i-len_row])
            if i>len_row+len_col:
                y=trans[i-len_row-len_col](x)

            if matrix_order(x, y, x0) == -1:  # In this case y>x
                x = y
                i = 0
            else:
                i = i + 1
        max_found=True
    return x

def gradient_ascent_seeded(x: np.array, x0='Daniel', gen_type='neighbourtranspositions'):

    '''Return the maximum of the gradient ascent with different initial seeds. The seeds matrices are produced by applying cyclic permutations to the start matrix.
    :param x: start value for the algorithm, which optimises along the group orbit of x
    :param x0: fixed matrix, the algorithm maximises the inner product with x0. A special case is x0=='Daniel', which gives a faster way to perform lexicographical ordering
    :param gen_type: the type of generators for the group action. Allowed types are 'neighbourtranspositions','alltranspositions','sudoku'
     '''

##run gradient_ascent for all seeds##
    k, m = np.shape(x)
    seeded_ascents=np.array([[gradient_ascent(np.dot(np.dot(pmatrix_cycle(i, k), x), pmatrix_cycle(j, m)), x0, gen_type=gen_type) for i in range(k)] for j in range(m)])

##find which seed has achieved the maximum inner product##
    k, m, n, l = np.shape(seeded_ascents)
    seeded_ascents_flattened = np.reshape(seeded_ascents, (k * m, n, l))

    if x0=='Daniel':
    #Here we are faced with the task to find the maximum of a list for a non-standard ordering. cmp_to_key allows for self defined orderings
        seeds_sorted=sorted(seeded_ascents_flattened, key=cmp_to_key(lambda x,y: matrix_order(x,y,x0)))
        #this is slightly wasteful, finding the maximum value of a list of length n takes (n-1) comparisons, while      sorting a list has complexity n*log(n)
        return seeds_sorted[-1]

    else:
        norms = [matrix_innerproduct(A,x0) for A in seeded_ascents_flattened]
        return seeded_ascents_flattened[np.argmax(norms)]



def dirichlet_project(matrix: np.array) -> np.array:
    '''Compute the dirichlet projection for a matrix of arbitrary size.
    '''
    return gradient_ascent_seeded(matrix)


if __name__ == '__main__':
    testlist=np.random.randint(6,size=(8,4,4))
    for A in testlist:
        A_proj = dirichlet_project(matrix=A)
        print('original matrix \n '+str(A)+'\n comb_ord \n '+str(A_proj)+ '\n')
