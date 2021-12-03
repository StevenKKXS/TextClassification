import os
import random
from tqdm import tqdm
import re
import pandas as pd
from nltk.stem import SnowballStemmer
import time
import numpy as np
from corpus import Corpus


class NaiveBayes(Corpus):
    def __init__(self, dataset_path='dataset', stop_words_path='stop_words_base.txt'):
        super(NaiveBayes, self).__init__(dataset_path, stop_words_path)
        self._word_freq = np.zeros((1, 1), dtype=int)
        self._corpura = []
        self._corpura_map = {}
        self._prob_cond = np.zeros((1, 1), dtype=float)
        self._tag_freq = np.zeros((1, 1), dtype=int)
        self._stemmer = SnowballStemmer('english')
        self._predict_record = []

    def pre_process_doc_freq(self, pbar_desc=''):
        # init
        self._corpura.clear()
        self._corpura_map.clear()
        self._predict_record.clear()
        # 获取所有词汇以及其并赋予对应位置
        train_words = set()
        train_words_stopworded = set()
        pbar = tqdm(total=len(self._train_set), desc=pbar_desc, ncols=100)
        for file in self._train_set:
            doc_tag = file.split('\\')[-2]
            # 记录未stem词语，方便train
            if self._corpura_map.get(doc_tag) is None:
                self._corpura_map.update({doc_tag: len(self._corpura)})
                self._corpura.append([0, {}])
            tag_pos = self._corpura_map[doc_tag]
            with open(file, 'r', encoding='ISO-8859-1') as f:
                f.readline()
                # 去除文件头
                # 去除前n行含有':'的和空行
                line = f.readline()
                while ':' in line or line == '\n':
                    line = f.readline()

                while not line == '':
                    words = self.get_words(line)
                    for word in words:
                        train_words.add(word)
                        if self._corpura[tag_pos][1].get(word) is None:
                            self._corpura[tag_pos][1].update({word: 1})
                        else:
                            word_value = self._corpura[tag_pos][1][word]
                            self._corpura[tag_pos][1].update({word: word_value + 1})
                    line = f.readline()

                # 记录每一类中的文档个数
                self._corpura[tag_pos][0] += 1
            pbar.update(1)
        # 统一stemmer，加快速度
        # 为了np.zeros初始化做准备，必须知道stem词的个数
        for word in train_words:
            stem_word = self._stemmer.stem(word)
            if stem_word not in self._stop_words:
                train_words_stopworded.add(stem_word)
        self._corpus_map = dict(zip(list(train_words_stopworded), [i for i in range(len(train_words_stopworded))]))
        # print(len(self._corpus_map))
        # print(self._corpus_map.get('contain'))

        self._tags_map = self._corpura_map

    def pre_process(self, pbar_desc=''):
        # init
        self._corpura.clear()
        self._corpura_map.clear()
        self._predict_record.clear()
        # 获取所有词汇以及其并赋予对应位置
        train_words = set()
        train_words_stopworded = set()
        pbar = tqdm(total=len(self._train_set), desc=pbar_desc, ncols=100)
        for file in self._train_set:
            doc_tag = file.split('\\')[-2]
            # 记录未stem词语，方便train
            if self._corpura_map.get(doc_tag) is None:
                self._corpura_map.update({doc_tag: len(self._corpura)})
                self._corpura.append([0, {}])
            tag_pos = self._corpura_map[doc_tag]
            with open(file, 'r', encoding='ISO-8859-1') as f:
                f.readline()
                # 去除文件头
                # 去除前n行含有':'的和空行
                line = f.readline()
                while ':' in line or line == '\n':
                    line = f.readline()

                while not line == '':
                    words = self.get_words(line)
                    for word in words:
                        train_words.add(word)
                        if self._corpura[tag_pos][1].get(word) is None:
                            self._corpura[tag_pos][1].update({word: 1})
                        else:
                            word_value = self._corpura[tag_pos][1][word]
                            self._corpura[tag_pos][1].update({word: word_value + 1})
                    line = f.readline()

                # 记录每一类中的文档个数
                self._corpura[tag_pos][0] += 1
            pbar.update(1)
        # 统一stemmer，加快速度
        # 为了np.zeros初始化做准备，必须知道stem词的个数
        for word in train_words:
            stem_word = self._stemmer.stem(word)
            if stem_word not in self._stop_words:
                train_words_stopworded.add(stem_word)
        self._corpus_map = dict(zip(list(train_words_stopworded), [i for i in range(len(train_words_stopworded))]))
        # print(len(self._corpus_map))
        # print(self._corpus_map.get('contain'))

        self._tags_map = self._corpura_map

    def train(self, pbar_desc=''):
        # 词向量多记录一行，最后一行记录未出现词的个数
        self._word_freq = np.zeros((len(self._corpus_map) + 1, len(self._tags_map)), dtype=int)
        self._prob_cond = np.zeros((len(self._corpus_map) + 1, len(self._tags_map)), dtype=float)

        pbar = tqdm(total=sum([len(a_corp[1]) for a_corp in self._corpura]), desc=pbar_desc, ncols=100)
        # word count
        for tag in self._corpura_map:
            tag_pos = self._corpura_map[tag]
            for word in self._corpura[tag_pos][1]:
                stem_word = self._stemmer.stem(word)
                if stem_word not in self._stop_words:
                    self._word_freq[self._corpus_map[stem_word], self._tags_map[tag]] += self._corpura[tag_pos][1][word]
                pbar.update(1)
        # P(c)
        self._tag_freq = np.array([corpus_ele[0] for corpus_ele in self._corpura])
        self._tag_freq = np.log(self._tag_freq / sum(self._tag_freq))
        # print(self._word_freq.sum())
        # print(self._tag_freq)

        # P(x_i | c_k)
        # print(self._word_freq.shape[0])
        for col in range(self._word_freq.shape[1]):
            not_zero_count = 0
            for count in self._word_freq[:, col]:
                if not count == 0:
                    not_zero_count += 1
            # print(list(self._tags_map)[col] + ' Zero_count : ' + str(self._word_freq.shape[0] - not_zero_count))
            self._prob_cond[:, col] = np.log((self._word_freq[:, col] + 1) / (
                    self._word_freq[:, col].sum() + 1 * self._word_freq.shape[0]))

        # print(self._word_freq.shape)
        # print(self._prob_cond)

    def predict(self, pbar_desc=''):
        pbar = tqdm(total=len(self._evaluate_set), desc=pbar_desc, ncols=100)
        for file in self._evaluate_set:
            # 记录文章的属性出现了多少次 ni0 ni1 ni2 ... ni(w-1) ni(w) 5 最后一个是未出现在词表词的个数
            doc_vector = np.zeros(self._word_freq.shape[0], dtype=int)
            doc_corpus = {}
            doc_corpus.clear()
            doc_ground_truth = file.split('\\')[-2]
            with open(file, 'r', encoding='ISO-8859-1') as f:
                f.readline()
                # 去除文件头
                # 去除前n行含有':'的和空行
                line = f.readline()

                while ':' in line or line == '\n':
                    line = f.readline()

                while not line == '':
                    words = self.get_words(line)
                    for word in words:
                        if doc_corpus.get(word) is None:
                            doc_corpus.update({word: 1})
                        else:
                            doc_corpus.update({word: doc_corpus[word] + 1})
                    line = f.readline()

            # 转化为向量
            for word in doc_corpus:
                stem_word = self._stemmer.stem(word)
                if self._corpus_map.get(stem_word) is None:
                    doc_vector[-1] += doc_corpus[word]
                else:
                    doc_vector[self._corpus_map[stem_word]] += doc_corpus[word]

            # predict calculate
            predict_vector = doc_vector.dot(self._prob_cond) + self._tag_freq
            # print(predict_vector)
            predict_tag = list(self._tags_map)[np.argmax(predict_vector)]
            # print(doc_ground_truth, predict_tag)
            if doc_ground_truth == predict_tag:
                self._predict_record.append(1)
            else:
                self._predict_record.append(0)
            pbar.update(1)
        return sum(self._predict_record) / len(self._predict_record)

    def k_cross_validation(self, k=5):
        precise_record = []
        self.shuffle_all_files()
        step = int(len(self._all_files) / k)
        for i in range(k):
            if i == k - 1:
                self._evaluate_set = self._all_files[step * i: len(self._all_files)]
                self._train_set = self._all_files[0: step * i]
            else:
                self._evaluate_set = self._all_files[step * i: step * (i + 1)]
                self._train_set = self._all_files[0: step * i] + self._all_files[step * (i + 1): len(self._all_files)]
            self.pre_process(pbar_desc='Round ' + str(i + 1) + ' pre_processing')
            self.train(pbar_desc='Round ' + str(i + 1) + ' training')
            precise_record.append(self.predict(pbar_desc='Round ' + str(i + 1) + ' predicting'))
        print(precise_record)
