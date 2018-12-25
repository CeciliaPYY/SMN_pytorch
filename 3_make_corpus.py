"""
1. 对清洗并分词之后的数据，建立词典；
2. 将训练数据和测试数据准备成模型需要的格式，并写入文件中；
"""
import os
import torch
import argparse
import logging
import sys 
import logging.handlers
import gc


def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
 
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def reps2idx(responses, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in rep] for rep in responses]


def uttes2idx(utterances, word2idx):
    return [[[word2idx[w] if w in word2idx else UNK for w in u] for u in utte] for utte in utterances]


def dealWithline(line,max_utte_len,max_cont_len):
    contexts = line.decode('utf-8').strip().split("\t")
    uttes, resp, l = contexts[1:-1], contexts[-1], contexts[0]
    resp = resp.split()
    uttes = [utte.split() for utte in uttes]

    # 截断模式
    if len(resp) > max_utte_len:
        resp = resp[:max_utte_len]



    if len(uttes) > max_cont_len:
        # close to response
        uttes = uttes[-max_cont_len:]


    for index, utte in enumerate(uttes):
        if len(utte) > max_utte_len:
            uttes[index] = utte[:max_utte_len]


    return resp,uttes,l


def getWD(filename,max_utte_len,max_cont_len,min_word_count,logger):

    wordCount = {
                '<pad>': 99999999,
                '<unk>': 99999999,
            }


    count = 0 
    for line in open(filename, "rb"):
        count +=1
        if count % 50000 == 0:
            logger.info('[getWD] reading {} rows'.format(count))
        
        resp,uttes,l = dealWithline(line,max_utte_len,max_cont_len)


        # 统计频数
        flag1 = 0
        for r in resp:
            flag1 += 1
            if flag1 % 1000 == 0:
                logger.info('{} words of responses has been added into dict!!'.format(flag1))
            if wordCount.get(r) != None:
                wordCount[r] += 1
            else:
                wordCount.setdefault(r, 1)
        flag2 = 0
        for u in uttes:
            flag2 += 1
            if flag2 % 1000 == 0:
                logger.info('{} words of utterances has been added into dict!!'.format(flag2))
            for w in u:
                if wordCount.get(w) != None:
                    wordCount[w] += 1
                else:
                    wordCount.setdefault(w, 1)

    # 按照频数排个序吧
    wordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

    logger.info('original dict length is {}'.format(len(wordCount)))


    filterDict = list(filter(lambda x: x[1]>min_word_count, wordCount))

    logger.info('Atfer filtering, the dict length is {}'.format(len(filterDict)))
    
    
    word2idx = dict(zip([ i[0] for i in filterDict] , list(range(len(filterDict)))))
    return word2idx


def word_idx_mapping(idx_file, org_file, word2idx, logger,max_utte_len,max_cont_len,rows_epoch=1000):
    max_utte_len,max_cont_len
    with open(idx_file + '_uttes.txt', "w") as idx_f_uttes , \
            open (idx_file + '_reps.txt', "w") as idx_f_reps ,\
                open (idx_file + '_label.txt', "w") as idx_f_labels :
        rows = 0
        for line in open(org_file, "rb"):
            rows += 1
            if rows % rows_epoch == 0:
                logger.info('[word_idx_mapping] {} sessions have been written into {} file'.format(rows, os.path.basename(org_file).split('.')[0]))
            resp, uttes, l = dealWithline(line,max_utte_len,max_cont_len)


            idx_f_labels.write(str(l))
            idx_f_labels.write('\n')
            idx_f_reps.write(str(reps2idx([resp], word2idx)[0]))
            idx_f_reps.write('\n')
            idx_f_uttes.write(str(uttes2idx([uttes], word2idx)[0]))
            idx_f_uttes.write('\n')   

            
def merge2Data(idx_file):
    utterances = []
    with open(idx_file + '_uttes.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            line = eval(line.strip('\n'))
            utterances.append(line)

    reponses = []
    with open(idx_file + '_reps.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            line = eval(line.strip('\n'))
            reponses.append(line)

    labels = []
    with open(idx_file + '_label.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            line = eval(line.strip('\n'))
            labels.append(line)
    return utterances, reponses, labels

def save_train_data(max_cont_len, max_utte_len, word2idx, train_idx_file, logger):
    
    train_utterances, train_reponses, train_labels = merge2Data(train_idx_file)
    logger.info('loading train')
    
    train_data = {
        'max_cont_len': max_cont_len,
        'max_utte_len': max_utte_len,
        'dict': {
            'dict': word2idx,
            'dict_size': len(word2idx.keys()),
        },
        'train': {
            'responses': train_reponses,
            'utterances': train_utterances,
            'labels': train_labels
        }
    }
    
#     torch.save(data, "data/corpus")
    logger.info("Train Data saved!!!!!!")
    return train_data

def save_test_data(max_cont_len, max_utte_len, word2idx, test_idx_file, logger):

    
    test_utterances, test_reponses, test_labels = merge2Data(test_idx_file)
    logger.info('loading train')
    
    test_data = {
        'max_cont_len': max_cont_len,
        'max_utte_len': max_utte_len,
        'dict': {
            'dict': word2idx,
            'dict_size': len(word2idx.keys()),
        },
        'test': {
            'responses': test_reponses,
            'utterances': test_utterances,
            'labels': test_labels
        }
    }
    
#     torch.save(data, "data/corpus")
    logger.info("Test Data saved!!!!!!")
    return test_data


def returnArgs():

    parser = argparse.ArgumentParser(
        description='parse args')

    parser.add_argument('--logdir', type=str, default='logdir')
    parser.add_argument('--data', type=str, default='./data/corpus')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=607)
    parser.add_argument('--lr', type=float, default=.001)

    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--emb_dim', type=int, default=200)
    parser.add_argument('--first_rnn_hsz', type=int, default=200)
    parser.add_argument('--fillters', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--match_vec_dim', type=int, default=50)
    parser.add_argument('--second_rnn_hsz', type=int, default=50)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    PAD = 0
    UNK = 1

    WORD = {
        PAD: '<pad>',
        UNK: '<unk>',
    }
    
    logger = setup_logger('MAIN', '.')
    
    
    # 设置基本参数
    filename = '../data/train_.txt'
    max_utte_len = 40 # 最大对话长度
    max_cont_len = 7 # 最长单句子长度
    min_word_count = 0
    
    train_idx_file = '../data/train_idx'
    test_idx_file = '../data/test_idx'
    train_org_file = '../data/train_.txt'
    test_org_file = '../data/valid_.txt'
    
    
    
    # 生成字典
    word2idx = getWD(filename, max_utte_len, max_cont_len, min_word_count, logger)

    torch.save(word2idx,'../data/word2idx')
    logger.info('word2idx build success !')
    

    word_idx_mapping(train_idx_file, train_org_file, word2idx,logger,max_utte_len,max_cont_len,10000)
    word_idx_mapping(test_idx_file, test_org_file, word2idx,logger,max_utte_len,max_cont_len,10000)
    logger.info('idx_file build success !')
    
    
    # 生成训练所需的训练数据和测试数据
    train_data = save_train_data(max_cont_len, max_utte_len, word2idx, train_idx_file, logger)
    logger.info('final train data build success !')
    torch.save(train_data, "../data/train_corpus")
    
    del train_data
    gc.collect()
    logger.info('remove train data ')
    
    
    test_data = save_test_data(max_cont_len, max_utte_len, word2idx, test_idx_file, logger)
    logger.info('final test data build success !')  
    torch.save(test_data, "../data/test_corpus")
    
    
