import numpy as np
import json
from projection_maps.auxiliary_functions.matrix_auxiliary_functions import permuteSingleMatrix, pad_matrix
import pickle
import os
from os.path import exists
from data.download_cicy_raw import download_cicy_raw


def findmatrix(text,start_index=0):
    '''In a string that comes from rawdata.txt, return the next matrix and its index is.
        :param text: the string to search for, only makes sense if it is a substring of rawdata.txt
        :param start_index: the index in the string from which we start the search
        :return: the next matrix and its final index in the string
    '''

    # Find the next matrix in the text beginning from start_index
    matrix_first_index = text.index('}\n{',start_index)
    matrix_last_index = text.index('}\n\n',start_index)
    matrix_string = text[matrix_first_index + 2:matrix_last_index]

    # convert into the standard python list of list notation
    matrix_string = matrix_string.replace('{', '[')
    matrix_string = matrix_string.replace('}\n', '],')
    matrix_string = '[' + matrix_string + ']]'

    # convert the matrix stored as a string into a python array
    matrix_string = np.array(json.loads(matrix_string))
    return matrix_string,matrix_last_index


def find_hodge(text: str,start_index=0):
    '''In a string that comes from rawdata.txt, return the next value of H11 and H21 as an integer.
    :param text: the string to search for, only makes sense if it is a substring of rawdata.txt
    :param start_index: the index in the string from which we start the search
    :return: two integers, the values of the next hodge numbers in the string
    '''

    # in the text file, the second hodge number always appears 8 characters after the 'H21' or 'H11'
    h1_first_index=text.index('H11',start_index)+8
    h1_last_index=h1_first_index+3

    h2_first_index=text.index('H21',start_index)+8
    h2_last_index=h2_first_index+3

    # We don't know whether the hodge number has 1 or 2 digits, so we might have to remove the last character
    while text[h1_last_index] == '\n' or text[h1_last_index]== 'H':
        h1_last_index = h1_last_index -1

    while text[h2_last_index] == '\n' or text[h2_last_index]== 'C':
        h2_last_index = h2_last_index -1

    h1 = int(text[h1_first_index:h1_last_index+1])
    h2 = int(text[h2_first_index:h2_last_index+1])

    return [h1,h2]

def create_cicy_original_pckl(n_perm=0, pad_with=0):
    '''Store a numpy array with all 7890 CICY matrices and one numpy array with the respective first and second hodge number in 'cicy_original.pckl'
    :param n_perm: number of additional permutations applied to each cicy matrix. If n_perm=0 then we recover the original cicy list
    :param pad_with: integer put on the main diagonal when the cicy matrix is written as an 12x15 matrix.
    :return: list of cicy matrices and a list of their hodge numbers
    '''
    if exists('cicy_original.pckl') == True:
        FileExistsError('The file cicy_original.pckl already exists and has not been created again.')
    else:
        download_cicy_raw()
        input_file_name = 'rawdata.txt'
        input_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{input_file_name}')
        f=open(input_filepath, "r") #data/rawdata.txt for windows and rawdata.txt for Ubuntu # using os.path works on all platforms
        contents=f.read()
        matrix_list=[]
        hodge_list=[]
        curr_ind = 0

        while curr_ind < len(contents)-3:

            hodge_numbers = find_hodge(contents,curr_ind)
            hodge_list.append(hodge_numbers)
            hodge_list = hodge_list + [hodge_numbers] * n_perm

            matrix, final_index = findmatrix(contents,curr_ind)
            matrix=pad_matrix(matrix, pad_with)
            matrix_list.append(matrix)
            permuted_matrices = list(map(permuteSingleMatrix, [matrix] * n_perm))
            matrix_list = matrix_list + permuted_matrices

            curr_ind = final_index + 1

        f.close()

        assert len(hodge_list) == 7890 and len(matrix_list)==7890

        output_file_name = 'cicy_original.pckl'
        output_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file_name)
        with open(output_filepath, 'wb') as f: # using os.path now works on all platforms
            pickle.dump([matrix_list, hodge_list], f)
