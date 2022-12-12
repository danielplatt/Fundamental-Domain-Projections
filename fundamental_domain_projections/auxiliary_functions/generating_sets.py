from fundamental_domain_projections.auxiliary_functions.matrix_permutation_functions import *

class generators:
    def __init__(self, name, matrix_dim, perm_axis):
        self.name = name
        self.matrix_dim = matrix_dim
        self.perm_axis = perm_axis
        self.elements = self.create_gen_set()

    def create_gen_set(self):
        k, m = self.matrix_dim
        assert self.name in ['neighbourtranspositions','alltranspositions','sudoku']

        if self.name == 'neighbourtranspositions':
            if self.perm_axis == 'row':
                generators_row = [transposition(l, l + 1, k) for l in range(k - 1)]
                generators_row.append(np.identity(k))
                return generators_row

            if self.perm_axis == 'col':
                generators_col = [transposition(l, l + 1, m) for l in range(m - 1)]
                return generators_col

            if self.perm_axis == 'trans':
                return []

        if self.name == 'alltranspositions':
            if self.perm_axis == 'row':
                generators_row = [transposition(i, j + 1, k) for j in range(k - 1) for i in range(j + 1)]
                generators_row.append(np.identity(k))
                return generators_row

            if self.perm_axis == 'col':
                generators_col = [transposition(i, j + 1, m) for j in range(m - 1) for i in range(j + 1)]
                return generators_col

            if self.perm_axis == 'trans':
                return []

        if self.name == 'sudoku':
            generators_both=[transposition(0,1,9),transposition(1,2,9),transposition(0,2,9),transposition(3,4,9),transposition(4,5,9),
                            transposition(3,5,9),transposition(6,7,9),transposition(7,8,9),transposition(6,8,9)]


            if self.perm_axis=='col':
                return generators_both

            if self.perm_axis=='row':
                generators_both.append(np.identity(9))
                return generators_both

            if self.perm_axis == 'trans':
                def transpose1(x):
                    return np.transpose(x)

                def transpose2(x):
                    return np.array([[x[8 - i, 8 - j] for i in range(0, 9)] for j in range(0, 9)])
                return [transpose1,transpose2]


if __name__ == '__main__':
    row_gen=generators(name='sudoku', matrix_dim=(9,9), perm_axis='trans')
    A=np.identity(9)
    A[0,0]=4
    A[1,0]=3
    print(A)
    print(row_gen.elements[1](A))