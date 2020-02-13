import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from tqdm import tqdm

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
# DEVELOPER_KEY='AIzaSyCI0jeuwxq8FSi_9ivFPJCjEX3hECTNNPA'
# DEVELOPER_KEY='AIzaSyBxwGcmJnkx8RUNDxO_6uH78mQ-ClEyHP0'
# DEVELOPER_KEY='AIzaSyACHEBX_lQNWOelQ2ziIEba2aNmHQgYwm0'
# DEVELOPER_KEY='AIzaSyA7_LlUY0dyvEE7poICo5ynlWZp1Z3xkCY'
# DEVELOPER_KEY='AIzaSyBF6RBo5Roc49yN8BvlG63Fc8h5BtGqTaY'
# DEVELOPER_KEY='AIzaSyDl8b1cvE2wDWJw4baPnQy20ctAff7jM4o'
# DEVELOPER_KEY='AIzaSyA09gpd9WA4l8QYlPvsroQ5PULq-28v6lc'
# DEVELOPER_KEY='AIzaSyAigH7jmDuJ7PnHcUFSB0VEAovUl18Ih-I'
DEVELOPER_KEY='AIzaSyBoT4n3eW0S92d05OeYE5Gmbde_OkJYvZw'
charset1 = sorted(list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')))
charset2 = sorted(list(set('AQgw')))

def createSubset():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    i = 0
    j = 0

    for cnt in tqdm(range(2000)):
        fp = open("subset.txt","a+")
        ls = []

        prefix = 'UC' + charset1[i] + charset1[j]
        search_response = youtube.search().list(q=prefix,part='id',maxResults=50,type='channel').execute()

        ls = search_response.get('items')
        if i != 63:
            i += 1
        else:
            i = 0
            j += 1
        if j == 64:
            j = 0
        for item in ls:
            d0 = item.get('id')
            d1 = d0.get('channelId')
            fp.write(d1+'\n')

        fp.close()

if __name__ == "__main__":
    createSubset()