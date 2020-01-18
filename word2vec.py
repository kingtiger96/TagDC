#encoding = utf-8

from gensim.models import Word2Vec
import pickle
from gensim.corpora.dictionary import Dictionary






def getSentences_list():
    sentence_words = []
    for i in range(1,47837):
        print(i)
        sentences = []
        sens = open("/data0/docker/lican/StackOverflowsmall/TitleBody/TitleBody%ld.txt"%i,encoding = 'utf-8').read().split()   
        for word in sens:
            sentences.append(word)
        print(sentences)
        sentence_words.append(sentences)
        
    print(len(sentence_words))   
    return sentence_words



def saveWordIndex(model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 词语的词向量
    pickle.dump(w2indx,open("/data0/docker/lican/StackOverflowsmall/cixiangliang1/w2indx.pkl", 'wb'))  # 索引字典
    print("w2indx")
    pickle.dump(w2vec,open("/data0/docker/lican/StackOverflowsmall/cixiangliang1/w2vec.pkl", 'wb'))  # 词向量字典
    print("w2vec")
    return w2indx, w2vec

def trainWord2Vec():#训练word2vec模型并存储
    sentences=getSentences_list()
    model=Word2Vec(sentences=sentences,size=256,sg=1,min_count=1,window=5)
    model.save('/data0/docker/lican/StackOverflowsmall/cixiangliang1/word2vec.model')
    print("word2vec")
    # model=Word2Vec.load('./word2vec.model')
    
    print(model.wv["c++"])
    
    saveWordIndex(model=model)



trainWord2Vec()

