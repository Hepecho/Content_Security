import cv2
import matplotlib.pyplot as plt
import os
from numpy import *

"""
hist = cv2.calcHist([img],  # 传入图像（列表）
                    [0],      # 使用的通道（使用通道：可选[0],[1],[2]）
                    None,     # 没有使用mask(蒙版)
                    [256],    # HistSize
                    [0, 256])  # 直方图柱的范围
# return->list
"""
color = ('b', 'g', 'r')


def hist_cal(image):
    hist_bgr = []
    for i, col in enumerate(color):
        hist_bgr.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
    return hist_bgr


def hist_show(hist_bgr):
    for i, col in enumerate(color):
        plt.plot(hist_bgr[i], color=col)
        plt.xlim([0, 256])
    plt.show()


def img_tranform(image):
    image = cv2.resize(image, (512, 512))
    return image


if __name__ == '__main__':
    # 示例
    img = cv2.imread('lena.jpg')
    hist_example = hist_cal(img)
    hist_show(hist_example)

    # 哆啦A梦和哆啦美
    src = cv2.imread('./data/source/dola.jpg')
    src = img_tranform(src)
    s_hist_rgb = hist_cal(src)

    hists = {}
    target_path = './data/target'
    names = os.listdir(target_path)
    # sys.exit()
    for name in names:
        child_path = os.path.join(target_path, name)
        # print(child_path)
        img = cv2.imread(child_path, 1)
        img = img_tranform(img)
        hist_rgb = hist_cal(img)
        hists[name] = hist_rgb

    # 按通道计算相似度
    print("巴氏距离")
    for k in hists:
        score = []
        print("compare with " + k)
        for i in range(3):
            score.append(cv2.compareHist(s_hist_rgb[i], hists[k][i], method=cv2.HISTCMP_BHATTACHARYYA))
            # 巴氏距离比较(method=cv.HISTCMP_BHATTACHARYYA) 值越小，相关度越高，最大值为1，最小值为0
        print("score_rgb: ", score)
        print("mean_score: ", mean(score))

    print("相关性")
    for k in hists:
        score = []
        print("compare with " + k)
        for i in range(3):
            score.append(cv2.compareHist(s_hist_rgb[i], hists[k][i], method=cv2.HISTCMP_CORREL))
            # 相关性比较 (method=cv.HISTCMP_CORREL) 值越大，相关度越高，最大值为1，最小值为0
        print("score_rgb: ", score)
        print("mean_score: ", mean(score))

    print("卡方比较")
    for k in hists:
        score = []
        print("compare with " + k)
        for i in range(3):
            score.append(cv2.compareHist(s_hist_rgb[i], hists[k][i], method=cv2.HISTCMP_CHISQR))
            # 卡方比较(method=cv.HISTCMP_CHISQR 值越小，相关度越高，最大值无上界，最小值0
        print("score_rgb: ", score)
        print("mean_score: ", mean(score))
