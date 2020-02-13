import os
import random

charset1 = sorted(list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')))
charset2 = sorted(list(set('AQgw')))

def pretreat():
    # fp = open("subset.txt","r")
    fp = open("data1.txt","r")
    content = fp.read()
    ls = content.split('\n')[0:-1]
    # charset = [set() for i in range(24)]  # find the pattern of ids
    # for i in range(0,24):
    #     for j in range(len(ls)):
    #         charset[i].update(ls[j][i])

    subset = {}
    for i in range(len(ls)):
        subset.update({ls[i]:i})
    newlist = list(subset)

    # pretreat2
    for i in range(len(newlist)):
        temp = 'UC'
        for j in range(2,23):
            rand = random.randint(0,99)
            prob = 80
            if newlist[i][j] != '-':
                prob = 50
            if rand < 50:
                ind = (charset1.index(newlist[i][j]) + rand) % 64
                newc = charset1[ind]
                temp += newc
            else:
                temp += newlist[i][j]
        temp += newlist[i][23]
        if temp != newlist[i]:
            # newlist[i] = temp
            newlist.append(temp)

    subset = {}
    for i in range(len(newlist)):
        subset.update({newlist[i]:i})
    newlist = list(subset)

    f1 = open("data1.txt","w")
    for i in range(len(newlist)):
        f1.write(newlist[i]+'\n')
    f1.close()
    fp.close()


def search(q):
    fp = open("data.txt","r")
    content = fp.read()
    ls = content.split('\n')[0:-1]
    ret = []
    l = len(q)
    q = q.lower()
    for i in range(len(ls)):
        temp = ls[i][0:l].lower()
        if temp == q:
            ret.append(ls[i])
    fp.close()
    return ret

pretreat()
