import random
import numpy as np

S = sorted(list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')))
N = 196849

def randSearch(m,L,ls):
    prefixset = []
    for i in range(m):
        prefix = 'UC'
        for j in range(L-2):
            rand = random.randint(0,63)
            prefix += S[rand]
        prefixset.append(prefix)
    retcnt = [0 for c in range(m)]
    for i in range(m):
        cnt = 0
        for st in ls:
            if st[2] < prefixset[i][2]:
                continue
            if st[2] > prefixset[i][2]:
                break
            if st[0:L] == prefixset[i]:
                cnt += 1
        retcnt[i] = cnt
    return sum(retcnt)

def estimate(sampleSum, num, m, L):
    prob_L = 1 / (num ** (L - 2))
    N_hat = (1 / (m * prob_L)) * sampleSum
    return N_hat

def rmse(prediction):
    return np.sqrt(((prediction - N) ** 2) / N)

if __name__ == "__main__":
    fp = open("data1.txt","r")
    content = fp.read()
    ls = content.split('\n')[0:-1]
    ls.sort()
    times = [4,5,6]
    for i in times:
        sampleSum = randSearch(30,i,ls)
        est = estimate(sampleSum,len(S),30,i)
        val = rmse(est)
        print("m=%d, L=%d, estimate=%f, rmse=%f"%(30,i,est,val))