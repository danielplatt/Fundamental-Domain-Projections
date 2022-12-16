from projection_maps.auxiliary_functions.matrix_auxiliary_functions import *

class generators:
    '''A class for generating sets for either row or column permutations or transpostions.
    :attribute gen_type: type of generating set, can be 'neighbourtranspositions', 'alltranspositions' or 'sudoku'
    :attribute matrix_dim: dimension of the matrix space the generators are applied on
    :attribute perm_axis: axis on which permutations act, can be 'row', 'col', or 'diagonal' (if the action is by matrix transpositions)
    '''
    def __init__(self, gen_type: str, matrix_dim: tuple, perm_axis: str):
        self.gen_type = gen_type
        self.matrix_dim = matrix_dim
        self.perm_axis = perm_axis
        self.elements = self.create_gen_set()

    def create_gen_set(self)->np.array:
        '''return a list of generators, stored as matrices, for an object of type generators'''
        k, m = self.matrix_dim
        assert self.gen_type in ['neighbourtranspositions', 'alltranspositions', 'sudoku']

        if self.gen_type == 'neighbourtranspositions':
            if self.perm_axis == 'row':
                generators_row = [pmatrix_trans(l, l + 1, k) for l in range(k - 1)]
                generators_row.append(np.identity(k))
                return generators_row

            if self.perm_axis == 'col':
                generators_col = [pmatrix_trans(l, l + 1, m) for l in range(m - 1)]
                return generators_col

            if self.perm_axis == 'diagonal':
                return []

        if self.gen_type == 'alltranspositions':
            if self.perm_axis == 'row':
                generators_row = [pmatrix_trans(i, j + 1, k) for j in range(k - 1) for i in range(j + 1)]
                generators_row.append(np.identity(k))
                return generators_row

            if self.perm_axis == 'col':
                generators_col = [pmatrix_trans(i, j + 1, m) for j in range(m - 1) for i in range(j + 1)]
                return generators_col

            if self.perm_axis == 'diagonal':
                return []

        if self.gen_type == 'sudoku':
            generators_both=[pmatrix_trans(0, 1, 9), pmatrix_trans(1, 2, 9), pmatrix_trans(0, 2, 9), pmatrix_trans(3, 4, 9), pmatrix_trans(4, 5, 9),
                             pmatrix_trans(3, 5, 9), pmatrix_trans(6, 7, 9), pmatrix_trans(7, 8, 9), pmatrix_trans(6, 8, 9)]


            if self.perm_axis=='col':
                return generators_both

            if self.perm_axis=='row':
                generators_both.append(np.identity(9))
                return generators_both

            if self.perm_axis == 'diagonal':
                def transpose1(x):
                    return np.transpose(x)

                def transpose2(x):
                    return np.array([[x[8 - i, 8 - j] for i in range(0, 9)] for j in range(0, 9)])
                return [transpose1,transpose2]


if __name__ == '__main__':
    row_gen=generators(gen_type='sudoku', matrix_dim=(9, 9), perm_axis='diagonal')
    A=np.identity(9)
    A[0,0]=4
    A[1,0]=3
    print(A)
    print(row_gen.elements[1](A))