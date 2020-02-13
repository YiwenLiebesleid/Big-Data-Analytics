import os
import random
from tqdm import tqdm

charset1 = sorted(list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')))
charset2 = sorted(list(set('AQgw')))

dataset = open("data1.txt","r").read().split('\n')[0:-1]

def search(q):  # return the number of matching ids in dataset
    ls = dataset
    ret = 0
    l = len(q)
    match = []
    for i in range(len(ls)):
        temp = ls[i][0:l]
        if temp == q:
            ret += 1
            match.append(ls[i])
    return ret

def prefixsearch(times,L):
    dict = {}
    for i in tqdm(range(times)):
        prefix = 'UC'
        for j in range(L-2):
            rand = random.randint(0, 63)
            prefix += charset1[rand]
        num = search(prefix)
        dict.update({prefix:num})
    return dict

if __name__ == "__main__":
    ret = prefixsearch(500,4)
    for val in ret.values():
        print(val)