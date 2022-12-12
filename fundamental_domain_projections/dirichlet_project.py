import numpy as np
from fundamental_domain_projections.auxiliary_functions.generating_sets import *
from fundamental_domain_projections.auxiliary_functions.matrix_permutation_functions import *
from functools import cmp_to_key #For user defined sorting

## The following algorithm can be used to approximate the 'Dirichlet projection map', see appendix E.2 of this
# preprint https://arxiv.org/pdf/2202.02164.pdf
# The element x0 is fixed and determines the ordering. If 'x0'==Daniel then it will be ordered lexicographically.
#The element x is the input matrix to which the algorithm is applied.
def gradient_ascent(x,x0,gen_name='neighbourtranspositions'):
    matrix_dim=np.shape(x)

# For matrix permutations there are three types of generators. In the CICY example, we don't use transpositions
    gen_row=generators(gen_name,matrix_dim,'row').elements
    gen_col=generators(gen_name,matrix_dim,'col').elements
    trans=generators(gen_name,matrix_dim,'trans').elements


    len_row=len(gen_row)
    len_col=len(gen_col)
    len_trans=len(trans)

    max_found=False
    while max_found==False:
        i=0
        while i<len_row+len_col+len_trans:
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

# We produce seed matrices by applying cyclic permutations to the start matrix and check which one achieves the highest value.

def gradient_ascent_seeded(x,x0, gen_name='neighbourtranspositions'):
    k, m = np.shape(x)
    seeded_ascents=np.array([[gradient_ascent(np.dot(np.dot(cycle(i,k),x),cycle(j,m)),x0, gen_name=gen_name) for i in range(k)]  for j in range(m)])
    if x0=='Daniel':
    #Here we are faced with the task to find the maximum of a list for a non-standard ordering. cmp_to_key allows for
    #self defined orderings
        seeds_sorted=sorted(flatten_onelevel(seeded_ascents), key=cmp_to_key(lambda x,y: matrix_order(x,y,x0)))
        return seeds_sorted[-1]
    else:
        array_of_norms=np.array([[matrix_innerproduct(seeded_ascents[j][i],x0) for i in range(k)] for j in range(m)])
        max_seed=argmax_nonflat(array_of_norms)
        return seeded_ascents[max_seed]


def dirichlet_project(matrix: np.array) -> np.array:
    '''TODO'''
    return gradient_ascent_seeded(A, x0='Daniel',gen_name='sudoku')


if __name__ == '__main__':
    #print(gradient_ascent_seeded(np.array([[0,1],[2,0]]),x0='Daniel'))
    A=np.array([[0,0,3,0,2,0,6,0,0],[9,0,0,3,0,5,0,0,1],[0,0,1,8,0,6,4,0,0],
                [0,0,8,1,0,2,9,0,0],[7,0,0,0,0,0,0,0,8],[0,0,6,7,0,8,2,0,0],
                [0,0,2,6,0,9,5,0,0],[8,0,0,2,0,3,0,0,9],[0,0,5,0,1,0,3,0,0]])
    print(A)
    print(gradient_ascent_seeded(A, x0='Daniel',gen_name='sudoku'))