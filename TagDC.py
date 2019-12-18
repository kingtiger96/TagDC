# coding=gbk
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
import keras.backend.tensorflow_backend as K
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.metrics import  precision_score ,recall_score
import heapq
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K1
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from sklearn.metrics import f1_score
import gensim
from scipy.spatial.distance import cosine
import os
import pandas
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import pandas


from gensim import corpora, models, similarities
import logging
from collections import defaultdict




# 参数设置
vocab_dim = 256 # 向量维度
maxlen = 100  # 文本保留的最大长度
batch_size = 512
n_epoch = 150
input_length = 100
  






def getSentences_list():
    sentences = []
    for i in range(1,47837):
        sens = open("/data/lican/StackOverflowsmall/TitleBody/TitleBody%ld.txt"%(i),encoding = 'utf-8').read().split()   
        sentences.append(sens)
    return sentences
   




def getLabels_list():
    labels = []
    for i in range(1,47837):
        labs = open("/data/lican/StackOverflowsmall/Tags_numpy/Tag_numpy%ld.txt"%(i),encoding = 'utf-8').read().split()   
        labels.append(labs)
    return labels
    


def labelProcess():
    labels = getLabels_list()
    labs = []
    for label in labels:
        label = list(map(int,label))
        labs.append(label)
    return labs    
  


def getsim_list():
    labels = []
    for i in range(4784):
        print(i)
        labs = open("/data/lican/StackOverflowsmall/similarity/similarity%ld.txt"%(i),encoding = 'utf-8').read().split()   
        labels.append(labs)
    return labels
    


def simProcess():
    labels = getsim_list()
    labs = []
    for label in labels:
        label = list(map(float,label))
        labs.append(label)
    return labs          



def text2index(index_dic,sentences):
   
    #把词语转换为数字索引,比如[['中国','安徽','合肥'],['安徽财经大学','今天','天气','很好']]转换为[[1,5,30],[2,3,105,89]]
    new_sentences=[]
    for sen in sentences:
        new_sen=[]
        for word in sen:
            try:
                new_sen.append(index_dic[word])
            except:
                new_sen.append(0)
        new_sentences.append(new_sen)
    return new_sentences


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
 
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(20, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)
 
    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
 
    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])
 
        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
 
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        #动态路由部分
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])
 
        return outputs
 
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


def attention_3d_block(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs]) 
    return output_attention_mul

 
class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        a = K.sqrt(K.sum(K.square(inputs), -1))
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train_lstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train, p_X_test, p_y_test):
    """
    :param p_n_symbols: word2vec训练后保留的词语的个数
    :param p_embedding_weights: 词索引与词向量对应矩阵
    :param p_X_train: 训练X
    :param p_y_train: 训练y
    :param p_X_test: 测试X
    :param p_y_test: 测试y0
    :return: 
    """
    print (u'chuangjian...')
    # build RNN model with attention
    inputs = Input(shape=(100,))

    embedding_output = Embedding(input_dim=p_n_symbols,
                                 output_dim=vocab_dim,
                                 mask_zero=False,
                                 weights=[p_embedding_weights],
                                 input_length=input_length,
                                 trainable=True,
                                 name = "Embedding")(inputs)
    em = Dropout(0.5)(embedding_output)
   
    x = Bidirectional(LSTM(units = 256,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   bias_initializer='zeros',
                   unit_forget_bias=True,
                   kernel_regularizer=None,
                   recurrent_regularizer=None, 
                   bias_regularizer=None, 
                   activity_regularizer=None, 
                   kernel_constraint=None,
                   recurrent_constraint=None,
                   bias_constraint=None, 
                   return_sequences = True,
                   dropout=0.2,
                   recurrent_dropout=0.0,
                   name = "Lstm"))(em)
    x = Concatenate()([x,em])
    x = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=256, kernel_size=1, padding="valid")(x)))
        
        
    x = Dropout(0.2)(x)
    
    x = Capsule(
    num_capsule=16,dim_capsule=16,
    routings=3, share_weights=True)(x)
    
    x = Length()(x)        
    """
    x = Flatten()(x)
        
        
    x = Dense(self.num_classes, activation = 'sigmoid')(x)
    """
    output_tensor = x

    model = Model([input_tensor], [output_tensor])


 


    model.summary()
    
    optimizer = Adam(lr=0.001)
    print (u'bianyi...')
    model.compile(loss=margin_loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
  

    print (u"xunlian...")
    checkpointer = ModelCheckpoint('chk_classifier.hdf5',monitor = 'val_loss', 
                       
                       verbose = 1, 
                       save_best_only = True, 
                       save_weights_only = True)
    early_stopper = EarlyStopping(monitor = 'val_loss', 
                       min_delta = 0.001, 
                       patience = 20)
    lr_reducer = ReduceLROnPlateau(monitor = 'val_loss',
                       factor = 0.1,
                       verbose = 1,
                       patience = 3,
                       min_lr = 2E-6)

    model.fit(p_X_train, p_y_train, batch_size=batch_size, epochs=n_epoch,validation_split=0.1,verbose=1, callbacks = [checkpointer, early_stopper, lr_reducer])
    
    """各层的结构和输入输出"""
    

    
    
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print ('Test score:', score)
    print ('Test accuracy:', acc)
    
 
    y_pred1 = model.predict(p_X_test)
    for i in range(0,len(y_pred1)):

        print(i)
        filename = "/data/lican/StackOverflowsmall/dl/Tag_numpy%d.txt"%(i+1)
        print(y_pred1[i])
        with open(filename,'w',encoding = 'utf-8',errors = 'ignore') as f:
            f.write(" ".join("%s"%id for id in y_pred1[i]))

    print(np.shape(y_pred1))
    y_prediction5 = []
    
    for i in range(len(y_pred1)):
        yy = []
        max_index = topk(y_pred1[i].tolist(),5)
        for p in range(len(y_pred1[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction5.append(yy)
        yy = []
    y_true = np.vstack(p_y_test)
    y_pred5 = np.vstack(y_prediction5)

    y_prediction10 = []
    
    for i in range(len(y_pred1)):
        yy = []
        max_index = topk(y_pred1[i].tolist(),10)
        for p in range(len(y_pred1[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction10.append(yy)
        yy = []
    y_pred10 = np.vstack(y_prediction10)

    print("top5")
    print("top5----precision_score:",precision_score(y_true, y_pred5,average = "samples"))
    print("top5----recall_score:",recall_score(y_true, y_pred5,average = 'samples'))
    #print(classification_report(y_true, y_pred5,target_names=tags)) 
    report1 = classification_report(y_true, y_pred5,output_dict=True)
    df = pandas.DataFrame(report1).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/dl5.xlsx')
    

    
    print("top10")
    print("top10----precision_score:",precision_score(y_true, y_pred10,average = "samples"))
    print("top10----recall_score:",recall_score(y_true, y_pred10,average = 'samples'))
    #print(classification_report(y_true, y_pred10))
    report2 = classification_report(y_true, y_pred10,output_dict=True)
    df = pandas.DataFrame(report2).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/dl10.xlsx')



def createModel():
    maxlen=100
 
    index_dict=pickle.load(open('/data/lican/StackOverflowsmall/cixiangliang/256/w2indx.pkl','rb'))
    
    vec_dict = pickle.load(open('/data/lican/StackOverflowsmall/cixiangliang/256/w2vec.pkl','rb'))
    
    n_words=len(index_dict.keys())
    print (n_words)
    vec_matrix=np.zeros((n_words+1,256))
    for k,i in index_dict.items():#将所有词索引与词向量一一对应
        try:
            vec_matrix[i,:]=vec_dict[k]
        except:
    
            print (k,i)
            print (vec_dict[k])
            exit(1)

    labels=labelProcess()
    sentences = getSentences_list()
    similaritylist = simProcess()

    
    X_train1 = sentences[:43052]
    X_test1 = labels[:43052]
    y_train1 = sentences[43052:]
    y_test1 = labels[43052:]
  

    for i in range(0,len(y_test1)):

        print(i)
        filename = "/data/lican/StackOverflowsmall/test/Tag_numpy%d.txt"%(i+1)
        print(y_test1[i])
        with open(filename,'w',encoding = 'utf-8',errors = 'ignore') as f:
            f.write(" ".join("%s"%id for id in y_test1[i]))
    print('done')

    X_train=text2index(index_dict,X_train1)
    X_test = text2index(index_dict, X_test1)
    
    print (u"xshape ", np.shape(X_train))
    print (u"xshape ", np.shape(X_test))
    

  
    y_train=np.array(y_train1)
    y_test =np.array(y_test1)
    
 
   
    print (u"yshape ", np.shape(y_train))
    print (u"yshape ", np.shape(y_test))
    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen,padding='post',truncating='post', value=0)#扩展长度不足的补0
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen,padding='post',truncating='post', value=0)
    print (u"shape ", np.shape(X_train))
    print (u"shape ", np.shape(X_test))
   
    print(np.shape(sentences))

    train_lstm(n_words+1, vec_matrix, X_train, y_train, X_test, y_test)

    scorelist = []
    for i in range(4784):
        score = scorecompute(similaritylist[i],y_train1,60)
        scorelist.append(score)
    print(scorelist)
    print(scorelist[1])

    for i in range(0,len(scorelist)):

        print(i)
        filename = "/data/lican/StackOverflowsmall/cf/Tag_numpy%d.txt"%(i+1)
        print(scorelist[i])
        with open(filename,'w',encoding = 'utf-8',errors = 'ignore') as f:
            f.write(" ".join("%s"%id for id in scorelist[i]))

    
    print(np.shape(scorelist))
    y_prediction5 = []
    
    for i in range(len(scorelist)):
        yy = []
        max_index = topk(scorelist[i].tolist(),5)
        for p in range(len(scorelist[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction5.append(yy)
        yy = []
    y_true = np.vstack(y_test1)
    y_pred5 = np.vstack(y_prediction5)

    y_prediction10 = []
    
    for i in range(len(scorelist)):
        yy = []
        max_index = topk(scorelist[i].tolist(),10)
        for p in range(len(scorelist[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction10.append(yy)
        yy = []
    y_pred10 = np.vstack(y_prediction10)
    print("top5")
    print("top5----precision_score:",precision_score(y_true, y_pred5,average = "samples"))
    print("top5----recall_score:",recall_score(y_true, y_pred5,average = 'samples'))
    #print(classification_report(y_true, y_pred5,target_names=tags)) 
    report1 = classification_report(y_true, y_pred5,output_dict=True)
    df = pandas.DataFrame(report1).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/cf5.xlsx')
    

    
    print("top10")
    print("top10----precision_score:",precision_score(y_true, y_pred10,average = "samples"))
    print("top10----recall_score:",recall_score(y_true, y_pred10,average = 'samples'))
    #print(classification_report(y_true, y_pred10,target_names=tags))
    report2 = classification_report(y_true, y_pred10,output_dict=True)
    df = pandas.DataFrame(report2).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/cf10.xlsx')
    
    


def topk(array_list,i):
   
    max_num_index_list = map(array_list.index, heapq.nlargest(i, array_list))
    return (list(max_num_index_list))





if __name__ == "__main__":
    createModel()
 

