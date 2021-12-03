import os
import random
from tqdm import tqdm
import re
import pandas as pd
from nltk.stem import SnowballStemmer
import time
import numpy as np
from corpus import Corpus
import heapq


def dict_dot_dict(d1, d2):
    score = 0
    if len(d1) < len(d2):
        for ele in d1:
            if not d2.get(ele) is None:
                score += d1[ele] * d2[ele]
    else:
        for ele in d2:
            if not d1.get(ele) is None:
                score += d2[ele] * d1[ele]
    return score


class K_NN(Corpus):
    def __init__(self, dataset_path='dataset', stop_words_path='stop_words_base.txt', para_k=3):
        super(K_NN, self).__init__(dataset_path, stop_words_path)
        self._para_k = para_k
        self._corpus_map = {}
        self._doc_count = np.zeros((1, 1), dtype=int)
        self._idf = np.zeros((1, 1), dtype=float)
        self._doc_pre_vectors = []
        self._doc_vectors = []
        self._doc_vectors_tfidf = []
        self._stemmer = SnowballStemmer('english')

    def set_k(self, k):
        self._para_k = k

    def pre_process(self, pbar_desc=''):
        # init
        self._doc_pre_vectors.clear()

        # 获取所有词汇以及其并赋予对应位置
        pbar = tqdm(total=len(self._train_set), desc=pbar_desc, ncols=100)
        for file in self._train_set:
            doc_tag = file.split('\\')[-2]
            doc_words = {}
            doc_words_stem = set()
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
                        if doc_words.get(word) is None:
                            doc_words.update({word: 1})
                        else:
                            doc_words.update({word: doc_words[word] + 1})
                    line = f.readline()

            # add pre_vector
            self._doc_pre_vectors.append([doc_tag, doc_words])
            pbar.update(1)
        pbar.close()
        # end
        # print(len(self._doc_pre_vectors))

    def train(self, pbar_desc=''):
        #init
        self._doc_vectors.clear()
        self._doc_vectors_tfidf.clear()
        self._corpus_map.clear()

        # stemming and stop_words
        pbar = tqdm(total=len(self._doc_pre_vectors), ncols=100, desc=pbar_desc)
        words_stemmed = set()
        for pre_vector in self._doc_pre_vectors:
            count_vector = {}
            for word in pre_vector[1]:
                stem_word = self._stemmer.stem(word)
                if stem_word not in self._stop_words:
                    words_stemmed.add(stem_word)
                    if count_vector.get(stem_word) is None:
                        count_vector.update({stem_word: pre_vector[1][word]})
                    else:
                        count_vector.update({stem_word: count_vector[stem_word] + pre_vector[1][word]})
            self._doc_vectors.append([pre_vector[0], count_vector])
            pbar.update(1)

        # init corpus_map 和 doc_count
        for i, word in enumerate(words_stemmed):
            self._corpus_map.update({word: i})
        self._doc_count = np.zeros(len(self._corpus_map), dtype=int)
        print(len(self._corpus_map))

        # word_doc_count
        for doc_vec in self._doc_vectors:
            for word in doc_vec[1]:
                self._doc_count[self._corpus_map[word]] += 1
        # idf
        # print(len(self._doc_vectors))
        self._idf = np.log(len(self._doc_vectors)) - np.log(self._doc_count + 1)
        pbar.close()

        # tf_idf vec
        for doc_vec in self._doc_vectors:
            doc_total_words = sum(list(doc_vec[1].values()))
            tfidf_vec = {}
            for word in doc_vec[1]:
                tfidf = doc_vec[1][word] / doc_total_words * self._idf[self._corpus_map[word]]
                tfidf_vec.update({word: tfidf})

            self._doc_vectors_tfidf.append([doc_vec[0], tfidf_vec])

        # print(len(self._doc_vectors_tfidf))

    def predict(self, pbar_desc=''):
        predict_record = []

        pre_process_count = 0
        pre_process_time = 0

        dict_dot_count = 0
        dict_dot_time = 0

        get_cand_time = 0
        get_cand_count = 0
        pbar = tqdm(total=len(self._evaluate_set), ncols=100, desc=pbar_desc)
        for file in self._evaluate_set:
            doc_ground_truth = file.split('\\')[-2]

            s_t = time.time()
            doc_vec_tfidf = self.get_doc_vec_tfidf(file)
            pre_process_time += time.time() - s_t
            pre_process_count += 1

            s_t = time.time()
            scores = [[vec[0], dict_dot_dict(vec[1], doc_vec_tfidf)] for vec in self._doc_vectors_tfidf]
            dict_dot_time += time.time() - s_t
            dict_dot_count += len(scores)

            s_t = time.time()
            candidates = heapq.nlargest(self._para_k, scores, key=lambda x: x[1])
            get_cand_time = time.time() - s_t
            get_cand_count += 1

            cand_count = {}
            for cand in candidates:
                if cand_count.get(cand[0]) is None:
                    cand_count.update({cand[0]: 1})
                else:
                    cand_count.update({cand[0]: cand_count[cand[0]] + 1})

            predict = ''
            max_count = -1
            for cand in cand_count:
                if cand_count[cand] > max_count:
                    max_count = cand_count[cand]
                    predict = cand

            result = 1 if predict == doc_ground_truth else 0
            predict_record.append(result)
            pbar.update(1)

        print('pre_process_time : ' + str(pre_process_time) + ' pre_process_count : ' + str(
            pre_process_count) + ' ' + str(pre_process_count / pre_process_time) + ' it/s')
        print('dict_dot_time : ' + str(dict_dot_time) + ' dict_dot_count : ' + str(
            dict_dot_count) + ' ' + str(dict_dot_count / dict_dot_time) + ' it/s')
        print('cand_time : ' + str(get_cand_time) + ' cand_count : ' + str(
            get_cand_count) + ' ' + str(get_cand_count / get_cand_time) + ' it/s')
        print(sum(predict_record) / len(predict_record))

        return sum(predict_record) / len(predict_record)

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

    def get_doc_vec_tfidf(self, file_path):
        # 必须在train 之后使用
        doc_vec_raw = {}
        # read and get raw_data
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            f.readline()
            # 去除文件头
            # 去除前n行含有':'的和空行
            line = f.readline()
            while ':' in line or line == '\n':
                line = f.readline()

            while not line == '':
                words = self.get_words(line)
                for word in words:
                    if doc_vec_raw.get(word) is None:
                        doc_vec_raw.update({word: 1})
                    else:
                        doc_vec_raw.update({word: doc_vec_raw[word] + 1})
                line = f.readline()

        # stemming and stop_words
        doc_vec = {}
        for word in doc_vec_raw:
            stem_word = self._stemmer.stem(word)
            if stem_word not in self._stop_words:
                if doc_vec.get(stem_word) is None:
                    doc_vec.update({stem_word: doc_vec_raw[word]})
                else:
                    doc_vec.update({stem_word: doc_vec[stem_word] + doc_vec_raw[word]})

        # tfidf
        vec_tfidf = {}
        total_words = sum(list(doc_vec.values()))
        for word in doc_vec:
            if not self._corpus_map.get(word) is None:
                vec_tfidf.update({word: doc_vec[word] / total_words * self._idf[self._corpus_map[word]]})
            # 注意到如果corpus里面没有word，那么就算平滑了之后，在之后点积也不会和对应项相乘，故直接去掉
        return vec_tfidf
