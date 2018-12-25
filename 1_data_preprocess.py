"""
1. 分别对训练数据和测试数据进行预处理（分词、去除停用词、去除无用的标点符号）；
"""



import pandas as pd
import nltk
import os
import re
from nltk.corpus import stopwords




def myown(x):
    tokenizer = nltk.tokenize.TweetTokenizer()
    tokenized_string = tokenizer.tokenize(x)
    return tokenized_string

def getdata(ele):
    split_ele = ele.split('\t')
    label = split_ele[0]
    ans   = split_ele[-1]
    utte  = split_ele[1:-1]
    return (label,ans,utte)
def removePunc(s):
    regex = re.compile(r"[\!\/,$%^*(+\"\']+|[+——！,?~@#￥%……&*()«»]+")
    s = regex.sub('', s)
    space = re.compile(r"\s+")
    return space.sub(' ',s)

def dealwithdata(filename, target_name rows):
     
    targetname = os.path.join(os.path.dirname(filename) , os.path.basename(filename).split('.')[0] + '_.txt')
    with open(target_name,'w',encoding = 'utf-8') as f :
        for  idx , row in enumerate( open(filename,'r',encoding='utf-8')):
            if idx >= rows:
                break
                l , a , u = getdata(row)

                a = myown(a)
                u =[ myown(ut) for ut in u]

                a = [  w.lower() for w in a ]
                u = [  [w.lower() for w in i ]  for i in u] 

                u = '<tab>'.join([' '.join(i) for i in u])
                a = ' '.join(a)


                u = removePunc(u)
                a = removePunc(a)

                u = u.replace('<tab>', '\t')

                f.write('{}\t{}\t{}\n'.format(l,u,a))


                if idx % 40000 == 0:
                    print(idx)
 
 if __name__ == "__main__":
    english_stopwords = stopwords.words("english")
    
    train_file  = '../data/train.txt'
    valfile    = '../data/valid.txt'

    target_train = '../data/train_.txt'
    target_test  = '../data/valid_.txt'

    dealwithdata(trainfile, 900000)
    dealwithdata(valfile, 10000)
