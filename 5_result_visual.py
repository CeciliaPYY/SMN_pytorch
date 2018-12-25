"""
1. 计算 TF-IDF 的表现情况，衡量标准为 recall@1, recall@2, recall@5；
2. 计算 Sequential Matching Model 的表现情况，衡量标准为 recall@1, recall@2, recall@5；
3. 比较二者的recall@1, recall@2, recall@5, 并画出比较的柱状图；
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
import torch
import easydict
from model import *
from data_loader import DataLoader
import os 
import csv
import random
import pickle
import scipy
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *


        
def recall(probas, k, group_size):   
    n_batches = len(probas) // group_size
    n_correct = 0
    for i in range(n_batches):
        batch = np.array(probas[i*group_size:(i+1)*group_size])
        #p = np.random.permutation(len(batch))
        #indices = p[np.argpartition(batch[p], -k)[-k:]]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
            
    return n_correct / (len(probas) / group_size)

def run(C_vec, R_vec, Y, group_size):
    batch_size = 10
    n_batches = len(Y) // 10
    probas = []
    YY = []
    recalls = dict.fromkeys(['recall@1', 'recall@2', 'recall@3'])

    for i in range(n_batches):
        if i % 10000 == 0:
            print(i)
        batch_c = C_vec[i*batch_size:(i+1)*batch_size][:group_size]
        batch_r = R_vec[i*batch_size:(i+1)*batch_size][:group_size]
        batch_y = Y[i*batch_size:(i+1)*batch_size][:group_size]
        YY.append(batch_y)
        probas += [1 - cosine(batch_c[0].toarray(), r.toarray()) for r in batch_r]
        
    c = 0
    for k in [1, 2, 5]:
        if k < group_size:
            recalls['recall@{}'.format(c)] = recall(probas, k, group_size)
            print('recall@%d: ' % k, recall(probas, k, group_size))
            
    return recalls

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))
        
    
if __name__ == "__main__":
    
    args = easydict.EasyDict()
    args.train_sample_ratio = 0.6
    args.max_cont_len = 7
    args.max_utte_len = 40
    args.use_cuda = True
    args.dict_size = len(word2idx.keys())
    args.emb_dim = 200
    args.first_rnn_hsz = 200
    args.second_rnn_hsz = 50
    args.kernel_size = 3
    args.fillters = 8
    args.match_vec_dim = 50
    args.test_batch_size = 10
    args.dropout = 0.5
    
    # 对比 TD-IDF 和 SMN 的 recall 结果
    
    # 1. SMN recall
    word2idx = torch.load('../../../data/word2idx')
    idx2word = { v:k for k , v in word2idx.items()}

    model = Model(args)
    model.load_state_dict(torch.load('./model/model_5.pkl')) 
    model.cuda()
#     model.eval()
    
    test_data = torch.load('../../../data/test_corpus')

    use_cuda = True
    validation_data = DataLoader(
            test_data['test']['utterances'],
            test_data['test']['responses'],
            test_data['test']['labels'],
            test_data['max_cont_len'],
            test_data['max_utte_len'],
            use_cuda,
            bsz=args.test_batch_size,
            shuffle=False,
            evaluation=True)
    
    model.eval()
    maxMatch = -1000
    maxPlace = None
    recalls = dict.fromkeys(['recall@1', 'recall@2', 'recall@3'])
    c = -1
    for K in [1, 2 ,5 ]:
        c += 1
        total = 0
        for u ,r , l in validation_data:
            pred = model(u, r)
            pred = pred.detach().cpu().numpy()[:,1]
            score  = pred[0]
            if maxMatch < score:
                maxPlace = (u,r,l)
                maxMatch = score
            pred.sort()
            total += np.argwhere(pred == score)[0][0] >= 10-K
        recalls['recall@{}'.format(c)] = total/1000.0
        print('recall@{} : {}'.format(K,total/1000.0))
    
    # 2. TF-IDF recall
    data = pickle.load(open('dataset.pkl','rb'))

    with open('W.pkl', 'rb') as f:
        W = pickle.load(f, encoding='latin1') 

    with open('W2.pkl', 'rb') as f:
        W2 = pickle.load(f, encoding='latin1') 

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f, encoding='latin1') 

    train = data[0]
    valid = data[1]
    test  = data[2]

    train_c ,train_r , train_y = train['c'] , train['r'] ,  train['y']
    valid_c , valid_r , valid_y = valid['c'] , valid['r'] , valid['y']
    test_c , test_r , test_y = test['c'] , test['r'] , test['y']
    
    train_sentence_c = [ ' '.join([ idx2word[i] for  i  in w ])   for w in train_c ]
    train_sentence_u = [ ' '.join([ idx2word[i] for  i  in w ])   for w in train_r ]
    valid_sentence_c = [ ' '.join([ idx2word[i] for  i  in w ])   for w in valid_c ]
    valid_sentence_u = [ ' '.join([ idx2word[i] for  i  in w ])   for w in valid_r ]
    test_sentence_c = [ ' '.join([ idx2word[i] for  i  in w ])   for w in test_c ]
    test_sentence_u = [ ' '.join([ idx2word[i] for  i  in w ])   for w in test_r ]
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_sentence_c+train_sentence_u+valid_sentence_c+valid_sentence_u)
    C_vec = vectorizer.transform(test_sentence_c)
    R_vec = vectorizer.transform(test_sentence_u)
    Y = np.array(test_y)
    tf_idf = run(C_vec, R_vec, Y, 10)



    # 3. 画出比较图
    name = ['1', '2', '3']
    total_width = 0.8
    n = 2

    width = total_width / n
    x = [i for i in range(3)]
    l1 = np.array(list(tf_idf.values()))
    l2 = np.array(list(smn.values()))


    a = plt.bar(x, l1, width = width , label = 'TF-IDF', fc = 'cyan')
    for i in range(len(x)):  
        x[i] = x[i] + width  
    b = plt.bar(x, l2, width=width, label='SMN',tick_label = name,fc = 'b')

    autolabel(a)
    autolabel(b)

    plt.xlabel('k')
    plt.ylabel('recall@k')
    plt.title('Recall Comparison between TF-IDF and SMN')
    plt.legend()
    plt.show()
