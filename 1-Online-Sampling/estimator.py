import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from tqdm import tqdm
import random

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

DEVELOPER_KEY1='AIzaSyDl8b1cvE2wDWJw4baPnQy20ctAff7jM4o'
DEVELOPER_KEY2='AIzaSyA09gpd9WA4l8QYlPvsroQ5PULq-28v6lc'

charset1 = sorted(list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')))
charset2 = sorted(list(set('AQgw')))

def estimator(m,L):
    if L > 23 or L <= 0:
        print("Input error!")
        return 0
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    est_cnt = 0
    if m > 80:
        youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY1)
        for i in range(80):
            prefix = 'UC'
            for j in range(L-2):
                rand = random.randint(0,63)
                prefix += charset1[rand]
            search_response = youtube.search().list(q=prefix, part='id', maxResults=50, type='channel').execute()
            ls = search_response.get('items')
            for it in ls:
                if it['id']['channelId'][0:L] == prefix:
                    est_cnt += 1
        m -= 80

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY2)
    for i in range(m):
        prefix = 'UC'
        for j in range(L - 2):
            rand = random.randint(0, 63)
            prefix += charset1[rand]
        search_response = youtube.search().list(q=prefix, part='id', maxResults=50, type='channel').execute()
        ls = search_response.get('items')
        for it in ls:
            if it['id']['channelId'][0:L] == prefix:
                est_cnt += 1

    est_ave = est_cnt / m
    prob_L = 1 / (len(charset1) ** (L - 2))
    N_hat = (est_ave / prob_L)
    return N_hat

if __name__ == "__main__":
    N_hat = estimator(m=160,L=7)
    print("m=160, L=7, N_hat=%f"%(N_hat))