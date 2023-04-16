import requests
from bs4 import BeautifulSoup
import bs4
import re


def example3():
    start_url = "http://www.whu.edu.cn/"
    find = False
    r = requests.get(start_url)
    r.encoding = r.apparent_encoding
    demo = r.text
    soup = BeautifulSoup(demo, 'html.parser')
    for tag in soup.find_all('a', string=re.compile('樱')):
        find = True
        print(tag.string)
    if not find:
        print("通知公告：")
        for i in range(110, 0, -1):
            if i == 110:
                url = start_url + 'tzgg.htm'
            else:
                url = start_url + 'tzgg/' + str(i) + '.htm'
            sub_r = requests.get(url)
            sub_r.encoding = sub_r.apparent_encoding
            sub_demo = sub_r.text
            sub_soup = BeautifulSoup(sub_demo, 'html.parser')
            for tag in sub_soup.find_all('a', string=re.compile('樱')):
                print(tag.string)


def example4():
    r = requests.get("https://top.baidu.com/board?tab=movie")
    r.encoding = 'usf-8'
    demo = r.text
    soup = BeautifulSoup(demo, 'html.parser')
    ulist = []
    print('序号\t片名')
    it = iter(soup.find_all('div', 'c-single-text-ellipsis'))
    for tag in it:
        ulist.append(tag.string)
        print(ulist.index(tag.string) + 1, ulist[ulist.index(tag.string)])
        par = tag.find_parent().find_parent()
        # print(par)
        msgs = iter(par.find_all('div', class_='intro_1l0wp'))
        for msg in msgs:
            print(msg.string)
        # next(it)  # 跳过影片简介
        print(" 简介："+next(it).text[:-8])


# 获取网页html文本
def getHTMLText(url):
    try:
        kv = {
            'user-agent': 'Mozilla/5.0'
        }
        r = requests.get(url, headers=kv, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


def fillUnivList(ulist, html):
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find('tbody').children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr('td')
            ulist.append([tds[0].text.replace('\n', '').replace(' ', ''), re.split(' ', tds[1].text)[1],
                          re.search(r'\w{2}', tds[2].text).group(0), re.search(r'\w{2}', tds[3].text).group(0),
                          tds[4].text.replace('\n', '').replace(' ', '')])


def printUnivList(ulist, num):
    tplt = "{0:^4}\t{1:{5}^12}\t{2:{5}^5}\t{3:{5}^5}\t{4:^10}"
    print(tplt.format("排名", "学校名称", "省市", "类型", "总分", chr(12288)))
    for i in range(num):
        u = ulist[i]
        print(tplt.format(u[0], u[1], u[2], u[3], u[4], chr(12288)))


def example5():
    uinfo = []
    url = 'https://www.shanghairanking.cn/rankings/bcur/2021'
    html = getHTMLText(url)
    fillUnivList(uinfo, html)
    printUnivList(uinfo, 20)  # 20 univs

def gamerank():
    r = requests.get("https://top.baidu.com/board?tab=game")
    r.encoding = 'usf-8'
    demo = r.text
    soup = BeautifulSoup(demo, 'html.parser')
    ulist = []
    print('序号\t游戏名')
    it = iter(soup.find_all('div', 'c-single-text-ellipsis'))
    for tag in it:
        ulist.append(tag.string)
        print(ulist.index(tag.string) + 1, ulist[ulist.index(tag.string)])
        par = tag.find_parent().find_parent()
        # print(par)
        msgs = iter(par.find_all('div', class_='intro_1l0wp'))
        for msg in msgs:
            print(msg.string)
        # next(it)  # 跳过简介
        print(" 简介：" + next(it).text[:-8])


if __name__ == '__main__':
    # example5()
    gamerank()
