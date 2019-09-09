from selenium import webdriver
from lxml import etree
import numpy as np
import re
from selenium.webdriver.support.ui import WebDriverWait  # 等待
from selenium.webdriver.support import expected_conditions as EC  # 条件设置
import selenium.webdriver.common.by as By
import requests
import pandas as pd
if __name__ == '__main__':
    all_data = pd.DataFrame()
    url = 'https://movie.douban.com/subject/26581837/comments?status=P'
    driver = webdriver.Chrome()
    driver.get(url)
    text = driver.page_source
    wait = WebDriverWait(driver, 10)
    for i in range(25):
        print(i)
        next_btn = wait.until(EC.element_to_be_clickable(
            (By.By.CSS_SELECTOR, '#paginator > a.next')
        ))  # 等待网页定位到对应元素，并将定位到的元素赋值
        text = driver.page_source
        dom = etree.HTML(text, etree.HTMLParser())
        user_names = dom.xpath('//*[@class="comment-info"]/a/text()')  # 用户名
        user_score = dom.xpath('//*[@class="comment-info"]/span[2]/@class')  # 评分
        comment_time = dom.xpath('//*[@class="comment-time "]/@title')  # 评论时间
        short = dom.xpath('//*[@class="short"]/text()')  # 评论正文
        user_url = dom.xpath('//*[@class="comment-info"]/a/@href')

        city = []
        for j in user_url:
            rq = requests.get(j)
            d = etree.HTML(rq.content,
                           etree.HTMLParser())
            city.append(d.xpath('//*[@id="profile"]/div/div[2]/div[1]/div/a/text()'))
            import time

            time.sleep(0.1)
        city = [i[0] if i != [] else '' for i in city]
        user_score = [int(re.findall('[0-9]{2}', i)[0]) for i in user_score]
        # 保存文件

        data = pd.DataFrame({'用户名': user_names, '评分': user_score,
                             '评论时间': comment_time, '短评': short,
                             '常居': city})
        all_data = pd.concat([all_data, data])
        all_data.to_excel('temp.xlsx')

        next_btn.click()  # 点击定位到的元素

