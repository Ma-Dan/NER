# -*- coding: utf-8 -*
import sys, pickle, os, random
import numpy as np
import collections
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import jieba

def build_word_index(word_embedding_file, src_vocab_file, tgt_file, tgt_vocab_file):
    '''
        生成单词列表，并存入文件之中。
    :return:
    '''
    if not os.path.exists(word_embedding_file):
        print('word embedding file does not exist, please check your file path.')
        return

    print('building word index...')
    if not os.path.exists(src_vocab_file):
        with open(src_vocab_file, 'w') as source:
            f = open(word_embedding_file, 'r')
            for line in f:
                values = line.split()
                word = values[0]  # 取词
                source.write(word + '\n')
        f.close()
    else:
        print('source vocabulary file has already existed, continue to next stage.')

    if not os.path.exists(tgt_vocab_file):
        with open(tgt_file, 'r') as source:
            dict_word = {}
            for line in source.readlines():
                line = line.strip()
                if line != '':
                    word_arr = line.split()
                    for w in word_arr:
                        dict_word[w] = dict_word.get(w, 0) + 1

            top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
            with open(tgt_vocab_file, 'w') as s_vocab:
                for word, frequence in top_words:
                    s_vocab.write(word + '\n')
    else:
        print('target vocabulary file has already existed, continue to next stage.')

def get_vocab_size(vocab_filename):
    '''
    :return: 训练数据中共有多少不重复的词。
    '''
    size = 0
    with open(vocab_filename, 'r') as vocab_file:
        for content in vocab_file.readlines():
            content = content.strip()
            if content != '':
                size += 1
    return size

def get_vocab(vocab_filename):
    '''
        获取识别类别表。
    :return:
    '''
    vocab = {}
    size = 0
    with open(vocab_filename, 'r') as vocab_file:
        for content in vocab_file.readlines():
            content = content.strip()
            if content != '':
                vocab[content] = size
                size += 1

    return vocab

def load_word2vec_embedding(word_embedding_file, embedding_dim, vocab_size):
    '''
        加载外接的词向量。
        :return:
    '''
    print('loading word embedding, it will take few minutes...')
    embeddings = np.random.uniform(-1,1,(vocab_size + 2, embedding_dim))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=(embedding_dim)))
    padding = np.asarray(rng.normal(size=(embedding_dim)))
    f = open(word_embedding_file)
    for index, line in enumerate(f):
        values = line.split()
        try:
            coefs = np.asarray(values[1:], dtype='float32')  # 取向量
        except ValueError:
            # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
            print(values[0], values[1:])

        embeddings[index] = coefs   # 将词和对应的向量存到字典里
    f.close()
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    '''return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, embedding_dim],
                           initializer=tf.constant_initializer(embeddings), trainable=False)'''
    return embeddings

def word2id(vocab, word, unknown):
    if word in vocab:
        return vocab[word]
    return unknown

def file2data(file, vocab, unknown):
    data = []
    f = open(file)
    for index, line in enumerate(f):
        values = line.split()
        line_data = []
        for value in values:
            line_data.append(word2id(vocab, value, unknown))
        data.append(line_data)
    f.close()

    return data

def read_corpus(src_file, tgt_file, src_vocab, tgt_vocab, src_unknown, tgt_unknown):
    src_data = file2data(src_file, src_vocab, src_unknown)
    tgt_data = file2data(tgt_file, tgt_vocab, tgt_unknown)

    data = []
    for i, (src, tgt) in enumerate(zip(src_data, tgt_data)):
        data.append([src, tgt])

    random.shuffle(data)

    return data

def batch_yield(data, batch_size):
    seqs, labels = [], []
    for (sent_, tag_) in data:
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(tag_)

    if len(seqs) != 0:
        yield seqs, labels

def pad_sequences(sequences, padding):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [padding] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def read_input(sentence, vocab, unknown):
    seq_list = []
    word_list = list(jieba.cut(sentence))
    line_data = []
    for word in word_list:
        line_data.append(word2id(vocab, word, unknown))
    seq_list.append(line_data)

    return seq_list, word_list

def label2tag(vocab, label):
    for tag in vocab:
        if vocab[tag] == label:
            return tag
    return ""

'''def create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id, tgt_unknown_id, share_vocab=False):
    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=src_unknown_id)
    if share_vocab:
      tgt_vocab_table = src_vocab_table
    else:
      tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=tgt_unknown_id)
    return src_vocab_table, tgt_vocab_table'''

'''class BatchedInput(collections.namedtuple("BatchedInput",
                                            ("initializer",
                                             "source",
                                             "target_input",
                                             "source_sequence_length",
                                             "target_sequence_length"))):
    pass

def get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, tag_padding, batch_size, buffer_size=None, random_seed=None,
                 num_threads=8, src_max_len=args.max_sequence, tgt_max_len=args.max_sequence, num_buckets=5):
    if buffer_size is None:
        # 如果buffer_size比总数据大很多，则会报End of sequence warning。
        # https://github.com/tensorflow/tensorflow/issues/12414
        buffer_size = batch_size * 10

    # src_dataset = tf.contrib.data.TextLineDataset(src_file)
    # tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)

    src_dataset = tf.data.TextLineDataset(args.src_file)
    tgt_dataset = tf.data.TextLineDataset(args.tgt_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # src_tgt_dataset = src_tgt_dataset.filter(
    #     lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_threads)
        src_tgt_dataset.prefetch(buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_threads)
        src_tgt_dataset.prefetch(buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in: (
            src, tgt_in, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(vocab_size+1,  # src
                            tag_padding,  # tgt_input
                            0,  # src_len -- unused
                            0))

    def key_func(unused_1, unused_2, src_len, tgt_len):
        if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
            bucket_width = 10

        bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(tf.contrib.data.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size
    ))
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, src_seq_len, tgt_seq_len) = (
        batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)'''
