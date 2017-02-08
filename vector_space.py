#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@version: python2.7.8 
@author: XiangguoSun
@contact: sunxiangguodut@qq.com
@file: vector_space.py
@time: 2017/2/7 17:29
@software: PyCharm
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# 引入Bunch类
from sklearn.datasets.base import Bunch
import cPickle as pickle#之前已经说过，不再赘述
from sklearn.feature_extraction.text import TfidfVectorizer#这个东西下面会讲

# 读取文件
def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

# 读取bunch对象
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

# 写入bunch对象
def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

#这个函数用于创建TF-IDF词向量空间
def vector_space(stopword_path,bunch_path,space_path):

    stpwrdlst = _readfile(stopword_path).splitlines()#读取停用词
    bunch = _readbunchobj(bunch_path)#导入分词后的词向量bunch对象
    #构建tf-idf词向量空间对象
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})
    '''
    在前面几节中，我们已经介绍了Bunch。
    target_name,label和filenames这几个成员都是我们自己定义的玩意儿，前面已经讲过不再赘述。
    下面我们讲一下tdm和vocabulary（这俩玩意儿也都是我们自己创建的）：
    tdm存放的是计算后得到的TF-IDF权重矩阵。请记住，我们后面分类器需要的东西，其实就是训练集的tdm和标签label，因此这个成员是
    很重要的。
    vocabulary是词向量空间的索引，例如，如果我们定义的词向量空间是（我，喜欢，相国大人），那么vocabulary就是这样一个索引字典
    vocabulary={"我":0,"喜欢":1,"相国大人":2}，你可以简单的理解为：vocabulary就是词向量空间的坐标轴，索引值相当于表明了第几
    个维度。
    我们现在就是要构建一个词向量空间，因此在初始时刻，这个tdm和vocabulary自然都是空的。如果你在这一步将vocabulary赋值了一个
    自定义的内容，那么，你是傻逼。
    '''

    '''
    与下面这2行代码等价的代码是：
    vectorizer=CountVectorizer()#构建一个计算词频（TF）的玩意儿，当然这里面不只是可以做这些
    transformer=TfidfTransformer()#构建一个计算TF-IDF的玩意儿
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    #vectorizer.fit_transform(corpus)将文本corpus输入，得到词频矩阵
    #将这个矩阵作为输入，用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵

    看名字你也应该知道：
    Tfidf-Transformer + Count-Vectorizer  = Tfidf-Vectorizer
    下面的代码一步到位，把上面的两个步骤一次性全部完成

    值得注意的是，CountVectorizer()和TfidfVectorizer()里面都有一个成员叫做vocabulary_(后面带一个下划线)
    这个成员的意义，与我们之前在构建Bunch对象时提到的自己定义的那个vocabulary的意思是一样的，相当于词向量
    空间的坐标轴。显然，我们在第45行中创建tfidfspace中定义的vocabulary就应该被赋值为这个vocabulary_

    他俩还有一个叫做vocabulary(后面没有下划线)的参数，这个参数和我们第45中讲到的意思是一样的。
    那么vocabulary_和vocabulary的区别是什么呢？
    vocabulary_：是CountVectorizer()和TfidfVectorizer()的内部成员，表示最终得到的词向量空间坐标
    vocabulary：是创建CountVectorizer和TfidfVectorizer类对象时，传入的参数，它是我们外部输入的空间坐标，不写的话，函数就从
    输入文档中自己构造。
    一般情况它俩是相同的，不一般的情况没遇到过。
    '''
    #构建一个快乐地一步到位的玩意儿，专业一点儿叫做：使用TfidfVectorizer初始化向量空间模型
    #这里面有TF-IDF权重矩阵还有我们要的词向量空间坐标轴信息vocabulary_
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
    '''
    关于参数，你只需要了解这么几个就可以了：
    stop_words:
    传入停用词，以后我们获得vocabulary_的时候，就会根据文本信息去掉停用词得到
    vocabulary:
    之前说过，不再解释。
    sublinear_tf:
    计算tf值采用亚线性策略。比如，我们以前算tf是词频，现在用1+log(tf)来充当词频。
    smooth_idf:
    计算idf的时候log(分子/分母)分母有可能是0，smooth_idf会采用log(分子/(1+分母))的方式解决。默认已经开启，无需关心。
    norm:
    归一化，我们计算TF-IDF的时候，是用TF*IDF，TF可以是归一化的，也可以是没有归一化的，一般都是采用归一化的方法，默认开启.
    max_df:
    有些词，他们的文档频率太高了（一个词如果每篇文档都出现，那还有必要用它来区分文本类别吗？当然不用了呀），所以，我们可以
    设定一个阈值，比如float类型0.5（取值范围[0.0,1.0]）,表示这个词如果在整个数据集中超过50%的文本都出现了，那么我们也把它列
    为临时停用词。当然你也可以设定为int型，例如max_df=10,表示这个词如果在整个数据集中超过10的文本都出现了，那么我们也把它列
    为临时停用词。
    min_df:
    与max_df相反，虽然文档频率越低，似乎越能区分文本，可是如果太低，例如10000篇文本中只有1篇文本出现过这个词，仅仅因为这1篇
    文本，就增加了词向量空间的维度，太不划算。
    当然，max_df和min_df在给定vocabulary参数时，就失效了。
    '''

    #此时tdm里面存储的就是if-idf权值矩阵
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary = vectorizer.vocabulary_

    _writebunchobj(space_path, tfidfspace)
    print "if-idf词向量空间实例创建成功！！！"

if __name__ == '__main__':
    stopword_path = "train_word_bag/hlt_stop_words.txt"#停用词表的路径
    bunch_path = "train_word_bag/train_set.dat"  #导入训练集Bunch的路径
    space_path = "train_word_bag/tfdifspace.dat"  # 词向量空间保存路径
    vector_space(stopword_path,bunch_path,space_path)