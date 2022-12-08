import numpy as np
import json
import pickle
from flatten_list import *
# from src import flatten_list
def padding(a, padwhat=1):
    if padwhat=='none':
        return a
    else:
        (c,r)=a.shape
        b=np.zeros((12,15))
        b[:c,:r]=a
        for i in range(c,12):
            # write ones or zeroes on the diagonal
            b[i,r+i-c]=padwhat
        return b

def findmatrix(text):
    a = text.index('}\n{')
    b = text.index('}\n\n')
    l = text[a + 2:b]
    l = l.replace('{', '[')
    l = l.replace('}\n', '],')
    l = '[' + l + ']]'
    l = np.array(json.loads(l))
    return [l,b]

def permuterow(a,p):
    return a[list(p)]
def permutecolumn(a,p):
    return permuterow(a.transpose(),p).transpose()
def addpermutations(a,n):
    return [permutecolumn(permuterow(a,np.random.permutation(12)),np.random.permutation(15)) for i in range(n)]

def findh1(text):
    a=text.index('H11')
    h1=text[a+9:a+12]
    while h1[-1] == '\n' or h1[-1]== 'H':
        h1=h1[0:len(h1)-1]
    return int(h1)

def findh2(text):
    a=text.index('H21')
    h2=text[a+9:a+12]
    while h2[-1] == '\n' or h2[-1] == 'C':
        h2=h2[0:len(h2)-1]
    return int(h2)

def return_cleaned_data(additionalperm=0,padwhat=1): #k is the added number of matrices with a row permutation, m the same for column perm
    f=open("rawdata.txt", "r") #data/rawdata.txt for windows and rawdata.txt for Ubuntu
    contents=f.read()
    matrixlist=[]
    hlist=[]
    while len(contents)>200:
        hlist.append([findh1(contents),findh2(contents)])
        matrix=padding(findmatrix(contents)[0],padwhat)
        matrixlist.append(matrix)
        contents=contents[findmatrix(contents)[1]+1:len(contents)]
    matrixlist.append(padding(np.array([[5]])))
    hlist.append([1,101])
    n = len(matrixlist)
    for j in range(n):
        matrixlist=matrixlist[:j]+addpermutations(matrixlist[j],additionalperm)+matrixlist[j:]
        hlist=hlist[0:j]+[hlist[j] for i in range(additionalperm)]+hlist[j:]

    return [matrixlist,hlist]

cleandata = return_cleaned_data(additionalperm=10,padwhat=0)
matrixlist=cleandata[0]
matrixlist=[flatten_list.flatten_list(A) for A in matrixlist] #This should be activated in the end but not for testing
cleandata[0]=matrixlist

f = open('cleandata_with_zeroes_in_matrixform_with_extra_permutations.pckl', 'wb')
#Windows: f = open('data/cleandata_with_zeroes.pckl', 'wb')
pickle.dump(cleandata, f)
f.close()

cleandata = return_cleaned_data(additionalperm=10,padwhat=1)
f = open('cleandata_with_ones_in_matrixform_with_extra_permutations.pckl', 'wb')
#Windows: f = open('data/cleandata_with_zeroes.pckl', 'wb')
pickle.dump(cleandata, f)
f.close()

cleandata=return_cleaned_data(additionalperm=0,padwhat='none')
f =  open('matrixform_notpadded_notpermuted.pckl', 'wb')
pickle.dump(cleandata, f)
f.close()