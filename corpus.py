import os
import random
from tqdm import tqdm
import re
import pandas as pd
from nltk.stem import SnowballStemmer
import time


class Corpus(object):
    def __init__(self, dataset_path='dataset', stop_words_path='stop_words_base.txt'):
        self._dataset_path = dataset_path
        self._stop_words = set()
        self._tags_map = {}
        self._corpus_map = {}
        self._all_files = []
        self._train_set = []
        self._evaluate_set = []
        self.load(self._dataset_path)



        # init stop_words
        with open(stop_words_path, 'r') as f:
            line = f.readline()
            while not line == '':
                self._stop_words.add(line.strip())
                line = f.readline()

    def load(self, dataset_path='dataset'):
        self._dataset_path = dataset_path
        for filepath, dirnames, filenames in os.walk(dataset_path):
            for filename in filenames:
                self._all_files.append(os.path.join(filepath, filename))

    def shuffle_all_files(self):
        random.shuffle(self._all_files)

    def gene_simple_train_set(self, evaluate_set_percent=0.2):
        self.shuffle_all_files()
        self._train_set = self._all_files[0:int(len(self._all_files) * 0.8)]
        self._evaluate_set = self._all_files[int(len(self._all_files) * 0.8):len(self._all_files)]

    def get_words(self, line):
        if line == '\n':
            return []
        line = line.strip().lower()
        # 去除含有数字的词
        line = re.compile(r'\S*[0-9]+\S*').sub('', line)

        # don't e-mail之类的词的提取
        con_str = re.compile(r'[a-z]+[\'-][a-z]+')
        # 's 可能是 is 或者 所有格，都是可以忽略的部分
        # 'll 可以忽略
        # 've 可以忽略
        # 'd 可以忽略
        ignore_one = re.compile('\'[sd]')
        ignore_two = re.compile('\'[vlra][elm]')
        words = con_str.findall(line)
        words = [ignore_one.sub('', word) for word in words]
        words = [ignore_two.sub('', word) for word in words]

        line = con_str.sub('', line)

        # 去除邮箱 网址
        line = re.compile(r'\S+[-@.]\S+').sub('', line)
        # 去除标点
        # 换成' '以防括号内部有文字
        line = re.compile(r'[!"#$%&()*+,\\./:;<=>?@\[\]^_`{|}~-]').sub(' ', line)
        line = re.compile('\'').sub(' ', line)
        # 去除空格外特殊空白符号
        line = re.compile('\s+').sub(' ', line)

        words = words + [ele for ele in line.strip().split(' ') if not ele == '']
        # 去除非常规字符
        cond_normal_letter = re.compile('.*[^a-z\-\']+')
        words = [word for word in words if cond_normal_letter.match(word) is None]
        return words


if __name__ == '__main__':
    corpus = Corpus()
    corpus.gene_simple_train_set()
