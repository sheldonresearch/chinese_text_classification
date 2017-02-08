#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@version: python2.7.8 
@author: XiangguoSun
@contact: sunxiangguodut@qq.com
@file: html_demo.py
@time: 2017/2/6 12:25
@software: PyCharm
"""
import sys
from lxml import html
# 设置utf-8 unicode环境
reload(sys)
sys.setdefaultencoding('utf-8')

def html2txt(path):
    with open(path,"rb") as f:
        content=f.read()
    r'''
    上面两行是python2.6以上版本增加的语法，省略了繁琐的文件close和try操作
    2.5版本需要from __future__ import with_statement
    新手可以参考这个链接来学习http://zhoutall.com/archives/325
    '''
    page = html.document_fromstring(content) # 解析文件
    text = page.text_content() # 去除所有标签
    return text

if __name__  =="__main__":
    # htm文件路径，以及读取文件
    path = "1.htm"
    text=html2txt(path)
    print text	 # 输出去除标签后解析结果