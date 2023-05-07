# coding=utf-8

import sys
import json
import base64
import time

import requests

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode

timer = time.perf_counter

API_KEY = '6g9xx94pUl2n2xnSAQyOrYsS'
SECRET_KEY = 'TPneDjwuRXVO30U8o1y6Bh8fnRg071qn'

# 需要识别的文件
AUDIO_FILE = './data/TAL_SER/wav/train/S0001/121051.wav'  # 只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
# 文件格式
FORMAT = AUDIO_FILE[-3:]  # 文件后缀只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式

CUID = '123456PYTHON'
# 采样率
RATE = 16000  # 固定值

# 普通版

DEV_PID = 1537  # 1537 表示识别普通话，使用输入法模型。根据文档填写PID，选择语言及识别模型
ASR_URL = 'http://vop.baidu.com/server_api'
SCOPE = 'audio_voice_assistant_get'  # 有此scope表示有asr能力，没有请在网页里勾选，非常旧的应用可能没有


# 测试自训练平台需要打开以下信息， 自训练平台模型上线后，您会看见 第二步：“”获取专属模型参数pid:8001，modelid:1234”，按照这个信息获取 dev_pid=8001，lm_id=1234
# DEV_PID = 8001 ;
# LM_ID = 1234 ;

# 极速版 打开注释的话请填写自己申请的appkey appSecret ，并在网页中开通极速版（开通后可能会收费）

# DEV_PID = 80001
# ASR_URL = 'http://vop.baidu.com/pro_api'
# SCOPE = 'brain_enhanced_asr'  # 有此scope表示有极速版能力，没有请在网页里开通极速版

# 忽略scope检查，非常旧的应用可能没有
# SCOPE = False

class DemoError(Exception):
    pass


"""  TOKEN start """

TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'


def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)

    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req)
        result_str = f.read()
    except URLError as err:
        print('token http response http code : ' + str(err.code))
        result_str = err.read()

    result_str = result_str.decode()

    # print(result_str)
    result = json.loads(result_str)
    # print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        # print(SCOPE)
        if not SCOPE in result['scope'].split(' '):
            raise DemoError('scope is not correct')
        # print('SUCCESS WITH TOKEN: %s  EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')


"""  TOKEN end """


def post_file(speech_url_list):
    task_id_list = []
    for speech_url in speech_url_list:
        url = 'https://aip.baidubce.com/rpc/2.0/aasr/v1/create'  # 创建音频转写任务请求地址
        body = {
            "speech_url": speech_url,
            "format": "pcm",  # 音频格式，支持pcm,wav,mp3
            "pid": 1537,  # 模型pid，1537为普通话输入法模型，1737为英语模型
            "rate": 16000  # 音频采样率，支持16000采样率
        }
        token = {"access_token": fetch_token()}
        headers = {'content-type': "application/json"}
        response = requests.post(url, params=token, data=json.dumps(body), headers=headers)
        # 返回请求结果信息，获得task_id，通过识别结果查询接口，获取识别结果
        print(response.text)
        result = json.loads(response.text)
        # print(result)
        task_id_list.append(result["task_id"])
    with open("task_ids.txt", "a") as of:
        of.write(str(task_id_list))


def query_res(task_id_list):
    """  发送查询结果请求 """
    results = []
    # 转写任务id列表，task_id是通过创建音频转写任务时获取到的，每个音频任务对应的值
    for task_id in task_id_list:
        url = 'https://aip.baidubce.com/rpc/2.0/aasr/v1/query'  # 查询音频任务转写结果请求地址
        body = {
            "task_ids": [task_id],
        }
        token = {"access_token": fetch_token()}
        headers = {'content-type': "application/json"}
        response = requests.post(url, params=token, data=json.dumps(body), headers=headers)
        print(json.dumps(response.json(), ensure_ascii=False))
        results.append(response.json())
    with open("results.txt", "w", encoding='utf-8') as of:
        for line in results:
            of.write(json.dumps(line, ensure_ascii=False))
            of.write('\n')

def read_id_list(path):
    f = open(path, "r", encoding='utf-8')
    lines = f.readlines()  # 读取所有行并返回列表
    for i in range(len(lines)):
        start = lines[i].find('\'')
        end = lines[i].rfind('\'')
        lines[i] = lines[i][start + 1: end]  # 只保留id
    print(lines)
    return lines

if __name__ == '__main__':
    speech_url_list = [
        "https://platform.bj.bcebos.com/sdk%2Fasr%2Fasr_doc%2Fdoc_download_files%2F16k.pcm"
    ]
    post_file(speech_url_list)
    # task_id_list = read_id_list('./task_ids.txt')
    # query_res(task_id_list)
