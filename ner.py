# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from args import args
from utils import get_logger
from data import build_word_index, get_vocab, load_word2vec_embedding, read_corpus, read_input, label2tag

## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

if __name__ == '__main__':

    build_word_index(args.word_embedding_file, args.src_vocab_file, args.tgt_file, args.tgt_vocab_file)

    src_vocab = get_vocab(args.src_vocab_file)
    src_vocab_size = len(src_vocab)
    src_unknown = src_vocab_size
    src_padding = src_vocab_size + 1
    #print(len(src_vocab))
    #print(vocab_size)

    tgt_vocab = get_vocab(args.tgt_vocab_file)
    tgt_vocab_size = len(tgt_vocab)
    tgt_unknown = tgt_vocab_size
    tgt_padding = tgt_vocab_size + 1
    #print(tgt_vocab)

    embedding = load_word2vec_embedding(args.word_embedding_file, args.embedding_dim, src_vocab_size)

    if args.mode == 'train':
        model = BiLSTM_CRF(args, embedding, src_vocab, tgt_vocab, src_padding, tgt_padding, paths)
        model.build_graph()
        train_data = read_corpus(args.src_file, args.tgt_file, src_vocab, tgt_vocab, src_unknown, tgt_unknown)
        print("train data: {}".format(len(train_data)))
        model.train(train_data=train_data, test_data=None)
    elif args.mode == 'demo':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embedding, src_vocab, tgt_vocab, src_padding, tgt_padding, paths)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('============= demo =============')
            saver.restore(sess, ckpt_file)
            while(1):
                print('Please input your sentence:')
                demo_sent = input()
                if demo_sent == '' or demo_sent.isspace():
                    print('See you next time!')
                    break
                else:
                    demo_data, demo_word = read_input(demo_sent, src_vocab, src_unknown)
                    #print(demo_data)
                    label = model.demo_one(sess, demo_data)
                    #print(label)
                    index = 0
                    for word in demo_word:
                        print(word + '(' + label2tag(tgt_vocab, label[0][index]) + ')')
                        index = index + 1
