from selenium import webdriver
from lxml import etree
import numpy as np
from selenium.webdriver.support.ui import WebDriverWait  # 等待
from selenium.webdriver.support import expected_conditions as EC  # 条件设置
import selenium.webdriver.common.by as By

name = []
time = []
comment = []
city = []
driver = webdriver.Chrome()
i = 0
url = 'https://movie.douban.com/subject/26581837/comments?start=' + str(i) + '&limit=20&sort=new_score&status=P'
driver.get(url)
text = driver.page_source
dom = etree.HTML(text, etree.HTMLParser())
wait = WebDriverWait(driver, 10)
next_bin = wait.until(EC.element_to_be_clickable(
    (By.By.CSS_SELECTOR, '#paginator > a.next')  # copy selector
))

next_bin.click()  # 点击翻页
for i in range(0, 500, 20):
    print(i)
    url = 'https://movie.douban.com/subject/26581837/comments?start=' + str(i) + '&limit=20&sort=new_score&status=P'
    driver.get(url)
    text = driver.page_source
    dom = etree.HTML(text, etree.HTMLParser())
    comment.append(dom.xpath('//*[@class="short"]/text()'))
    name.append(dom.xpath('//*[@class="comment-info"]/a/text()'))
    city.append(dom.xpath('//*[@class="comment-info"]/a/@href'))
    time.append((dom.xpath('//*[@class="comment-info"]/span[3]/@title')))
name = np.array(name).reshape((-1,))
time = np.array(time).reshape((-1,))
comment = np.array(comment).reshape((-1,))
city = np.array(city).reshape((-1,))
print(city.shape)
print(city)
