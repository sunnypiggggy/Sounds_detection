import os
import sys
import string
import glob
import numpy as np
import collections

if __name__ == '__main__':
    model_type='cnn'
    result_dirs=glob.glob(model_type+'*'+'perdiction'+'*.txt')
    results=[]
    for var in result_dirs:
        with open(var) as f:
            temp=f.readlines()
            temp=list(map(int,temp))
            results.append(np.asarray(temp))
    results=np.asarray(results)

    shape=results.shape

    vote_result=[]
    for i in range(shape[1]):
        occ_dict=collections.Counter(results[:,i])
        best_vote_result=sorted(occ_dict,key=occ_dict.get)[-1]
        vote_result.append(best_vote_result)

    with open(model_type+'_result.txt','w+') as f:
        for var in vote_result:
            f.write(str(var)+'\n')
    print("shit")