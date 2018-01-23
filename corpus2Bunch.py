#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@version: python2.7.8 
@author: XiangguoSun
@contact: sunxiangguodut@qq.com
@file: corpus2Bunch.py
@time: 2017/2/7 7:41
@software: PyCharm
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os#python内置的包，用于进行文件目录操作，我们将会用到os.listdir函数
import cPickle as pickle#导入cPickle包并且取一个别名pickle
'''
事实上python中还有一个也叫作pickle的包，与这里的名字相同了，无所谓
关于cPickle与pickle，请参考博主另一篇博文：
python核心模块之pickle和cPickle讲解
http://blog.csdn.net/github_36326955/article/details/54882506
本文件代码下面会用到cPickle中的函数cPickle.dump
'''
from sklearn.datasets.base import Bunch
#这个您无需做过多了解，您只需要记住以后导入Bunch数据结构就像这样就可以了。
#今后的博文会对sklearn做更有针对性的讲解


def _readfile(path):
    '''读取文件'''
    #函数名前面带一个_,是标识私有函数
    # 仅仅用于标明而已，不起什么作用，
    # 外面想调用还是可以调用，
    # 只是增强了程序的可读性
    with open(path, "rb") as fp:#with as句法前面的代码已经多次介绍过，今后不再注释
        content = fp.read()
    return content

def corpus2Bunch(wordbag_path,seg_path):
    catelist = os.listdir(seg_path)# 获取seg_path下的所有子目录，也就是分类信息
    #创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    '''
    extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充
    原来的list
    '''
    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = class_path + file_path  # 拼出文件名全路径
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(_readfile(fullname))  # 读取文件内容
            '''append(element)是python list中的函数，意思是向原来的list中添加element，注意与extend()函数的区别'''
    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print "构建文本对象结束！！！"

if __name__ == "__main__":#这个语句前面的代码已经介绍过，今后不再注释
    #对训练集进行Bunch化操作：
    wordbag_path = "train_word_bag/train_set.dat"  # Bunch存储路径
    seg_path = "train_corpus_seg/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：
    wordbag_path = "test_word_bag/test_set.dat"  # Bunch存储路径
    seg_path = "test_corpus_seg/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)