import requests
import pandas as pd
import json

url = "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv4&productId=41477510129&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1"
header = {"Referer": "https://item.jd.com/41477510129.html",
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
          }
all_data = pd.DataFrame()
for i in range(10):
    url = f"https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv4&productId=41477510129&score=0&sortType=5&page={i}&pageSize=10&isShadowSku=0&fold=1"
    # print(url)
    rq = requests.get(url, headers=header)
    # print(rq.text)
    txt = rq.text
    data = json.loads(
        txt[len('fetchJSON_comment98vv4('): -2]
    )
    # print(type(data))
    content = [i['content'] for i in data['comments']]
    nickname = [i['nickname'] for i in data['comments']]
    referenceName = [i['referenceName'] for i in data['comments']]
    referenceTime = [i['referenceTime'] for i in data['comments']]
    productColor = [i['productColor'] for i in data['comments']]
    data = pd.concat(
        [pd.DataFrame(content), pd.DataFrame(nickname), pd.DataFrame(referenceName), pd.DataFrame(referenceTime),
         pd.DataFrame(productColor)], axis=1)
    all_data = pd.concat([all_data, data])
    # print(all_data)
# print(data.head)
all_data.to_csv('./data.csv')
