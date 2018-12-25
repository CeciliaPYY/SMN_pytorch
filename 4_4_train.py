"""
1. 定义模型训练流程；
"""

import argparse
import torch
from model import Model
from data_loader import DataLoader
import logging
import sys
import os
from const import *
import gensim
import numpy as np
from torch.autograd import Variable



def arg_parse():
    
    parser = argparse.ArgumentParser(
        description='A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots')

    parser.add_argument('--logdir', type=str, default='logdir')
    parser.add_argument('--data_train', type=str, default='../../../data/train_corpus')
    parser.add_argument('--data_test', type=str, default='../../../data/test_corpus')
    parser.add_argument('--save_model_dir', type=str, default='./model')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=607)
    parser.add_argument('--lr', type=float, default=.001)
    
    parser.add_argument('--train_sample_ratio', type=float, default=0.6)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--emb_dim', type=int, default=200)
    parser.add_argument('--first_rnn_hsz', type=int, default=200)
    parser.add_argument('--fillters', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--match_vec_dim', type=int, default=50)
    parser.add_argument('--second_rnn_hsz', type=int, default=50)

    args = parser.parse_args()
    return args

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



def add_summary_value(key, value):
    global tf_step

    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    tf_summary_writer.add_summary(summary, tf_step)


def evaluate(logger):
    model.eval()
    #logger.info('--this is evaluate progress')
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    total_corrects = 0
    for utterances, responses, labels in validation_data:

        pred = model(utterances, responses)
        loss = criterion(pred, labels)
        eval_loss += loss.item()
        corrects = (torch.argmax(pred[:,1])==0)
        total_corrects +=corrects.item()
        
    return eval_loss / _size, total_corrects, total_corrects / (_size/10.0) , _size/10.0


    
    
if __name__ == '__main__':

    args = arg_parse()

    logger = setup_logger("Training Process", './train_log')

    torch.manual_seed(args.seed)
    args.use_cuda = use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(args.seed)


    train_data = torch.load(args.data_train)
    test_data = torch.load(args.data_test)

    args.max_cont_len = train_data["max_cont_len"]
    args.max_utte_len = train_data["max_utte_len"]
    args.dict_size = train_data['dict']['dict_size']
    args.kernel_size = (args.kernel_size, args.kernel_size)

    logger.info("=" * 30 + "arguments" + "=" * 30)
    for k, v in args.__dict__.items():
        if k in ("epochs", "seed", "data"):
            continue
        logger.info("{}: {}".format(k, v))
    logger.info("=" * 60)


    training_data = DataLoader(
             train_data['train']['utterances'],
             train_data['train']['responses'],
             train_data['train']['labels'],
             train_data['max_cont_len'],
             train_data['max_utte_len'],
             use_cuda,
             bsz=args.train_batch_size)

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


    from model import Model

    model = Model(args)
    #model.load_state_dict(torch.load('/data/mask/pytorch/data/text/ubuntu_project/src/torchCode/retrieval-based-chatbots/model/model_50.pkl'))

    word2idx = torch.load('../../../data/word2idx')
    idx2word = { v:k for k , v in word2idx.items()}

    logging.info('pad has idx {}'.format(word2idx['<pad>']))
    logging.info('unk has idx {}'.format(word2idx['<unk>']))
    
    
    logger.info('loading pretrained word vector')
    glove_file=  '/data/mask/pytorch/data/text/glove.6B.200d.txt'
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(glove_file, binary=False, encoding='utf-8')
    logging.info('load success ')
    vocab_size = args.dict_size
    embed_size = 200
    weight = torch.zeros(vocab_size, embed_size)

    for i in range(len(wvmodel.index2word)):
        if i % 10000 == 0:
            print(i)
        try:
            index = word2idx[wvmodel.index2word[i]]
            print(i,word2idx[wvmodel.index2word[i]] , wvmodel.index2word[i] )
        except:
            continue
        break
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx2word[word2idx[wvmodel.index2word[i]]]))
    model.emb.data = weight


    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()


    try:
        logger.info('-' * 90)
        for epoch in range(1, args.epochs + 1):


            model.train()
            for  idx , (utterances, responses, labels) in enumerate( training_data ):
                optimizer.zero_grad()

                pred = model(utterances, responses)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                if idx % 5000 == 0 :
                    loss, corrects, acc, size = evaluate(logger)
                    logger.info('| idx {}  of epoch {:3d} | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(
                        idx , epoch, loss, acc, corrects, size))
                    torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'model_{}.pkl'.format(epoch)))
                    model.train()  

    except KeyboardInterrupt:
        logger.info("-" * 90)
        logger.info("Exiting from training early")
