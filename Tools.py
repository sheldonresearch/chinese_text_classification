#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:chinese_text_classification
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: Tools.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/01/23 
"""
import pickle


# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch
