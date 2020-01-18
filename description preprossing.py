# -*- coding: utf-8 -*-
"""Spyder Editor

This is a temporary script file.
"""




import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

"""停用词表更新"""
stopwords_list = []
stop_words = open("/home/lican/docker/stopwords.txt",encoding = 'utf-8').read()
stopwords_list = stop_words.split("\n")


p1 = re.compile(r'[\w]+')
def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False



for i in range(1,104045):
    
    raw = open("/data0/docker/lican/English/TitleBody/TitleBody(raw)/TitleBody%ld.txt"%(i),encoding = 'utf-8').read().lower()   #（依次读取目标文件已经解析好的数据集）
    titlebody = re.findall(p1,raw)
    
    filtered_words = [word for word in titlebody if word not in stopwords_list]   #去除停用词
    
    porter_stemmer = PorterStemmer()         #使用nltk.stem.porter的PorterStemmer方法提取单词的主干
    stemmer_words = [porter_stemmer.stem(word) for word in filtered_words]


    #print('stemmer--------', stemmer_words)
    print(i)
    TitleBody = []
    for word in stemmer_words:
        panduan = is_number(word)
        if panduan == False:
            TitleBody.append(word)    
    
    #print(TitleBody)
    filename = "/data0/docker/lican/English/TitleBody/TitleBody/TitleBody%ld.txt"%i  
    
    
    with open(filename,'w',encoding = 'utf-8') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(" ".join(TitleBody))
     
       
 

