#!/usr/bin/env python

from word2vec import *
import logging

logging.basicConfig(level=logging.INFO)

def run(text_path = 'text8', vocab_path = 'vocab.txt', output_path = 'vectors.txt'):

    vocab = Vocabulary(vocab_path)

    logging.info(f'字典中word_index映射关系建立完成')

    vocab.corpusWordsFunc(text_path)

    logging.info(f'语料加载完成')

    vocab.freqCorpusWords()

    logging.info(f'词频统计完成')

    vocab.buildNegativeTable()

    logging.info(f'负采样词表制作完成')

    net = network(vocab, embedDimension, contextSize, learningRate, negativeSampleSize, epochs)
    net.buildTable()

    saveFile(vocab.vocabWords, net.layer1, output_path)

run()