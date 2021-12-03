import os
from tqdm import tqdm
import re
import pandas as pd
from nltk.stem import SnowballStemmer
import time


def get_valid_words(line):
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
    stemmer = SnowballStemmer('english')
    return words

def corpus_freq_raw_gene():
    dir_path = r'dataset\20_newsgroups'
    corpus = {}
    count = 0
    with tqdm(total=19997) as pbar:
        for sub_dir in os.listdir(dir_path):
            group_corpus = {}
            for f_name in os.listdir(dir_path + '\\' + sub_dir):
                doc_corpus = {}
                f_path = dir_path + '\\' + sub_dir + '\\' + f_name
                with open(f_path, 'r', encoding='ISO-8859-1') as f:
                    # 去除文件头
                    # 去除前n行含有':'的和空行
                    line = f.readline()
                    while ':' in line or line == '\n':
                        line = f.readline()

                    while not line == '':
                        words = get_valid_words(line)
                        # 统计词频
                        cond_normal_letter = re.compile('[^a-z\-\']+')
                        for word in words:
                            # 去除特殊字符
                            if not cond_normal_letter.match(word) is None:
                                # print('f_path=' + f_path)
                                # print('word=' + word)
                                continue
                            if doc_corpus.get(word) is None:
                                doc_corpus.update({word: 1})
                            else:
                                doc_corpus.update({word: doc_corpus.get(word) + 1})
                        line = f.readline()
                # doc_corpus 更新到 group_corpus
                for word in doc_corpus:
                    if group_corpus.get(word) is None:
                        group_corpus.update({word: [doc_corpus.get(word), 1]})
                    else:
                        word_value = group_corpus.get(word)
                        group_corpus.update({word: [word_value[0] + doc_corpus.get(word), word_value[1] + 1]})

                # 进度条更新
                pbar.update(1)
                count = count + 1

            # group_corpus 更新到 corpus
            for word in group_corpus:
                if corpus.get(word) is None:
                    update_value = group_corpus.get(word)
                    corpus.update({word: [update_value[0], update_value[1], 1]})
                else:
                    update_value = group_corpus.get(word)
                    word_value = corpus.get(word)
                    corpus.update(
                        {word: [word_value[0] + update_value[0], word_value[1] + update_value[1], word_value[2] + 1]})

        # csv_gene
        pd.DataFrame([[word, corpus[word][0], corpus[word][1], corpus[word][2]] for word in corpus],
                     columns=['word', 'word_freq', 'doc_freq', 'group_freq']).to_csv('corpus_freq_snowball_raw.csv')
    print(len(corpus))
    print(count)


def corpus_freq_gene(corpus_raw_path='corpus_freq_snowball_raw.csv', stop_word_path='stop_words_base.txt'):
    corpus_raw = pd.read_csv(corpus_raw_path, keep_default_na=False)
    corpus = []
    stop_word = set()
    with open(stop_word_path, 'r') as f:
        line = f.readline()
        while not line == '':
            stop_word.add(line.strip())
            line = f.readline()

    with tqdm(total=len(corpus_raw['word'])) as pbar:
        for i, word in enumerate(corpus_raw['word']):
            if word not in stop_word:
                corpus.append([word, corpus_raw['word_freq'][i], corpus_raw['doc_freq'][i], corpus_raw['group_freq'][i]])
            pbar.update(1)

    # csv_gene
    pd.DataFrame(corpus,
                 columns=['word', 'word_freq', 'doc_freq', 'group_freq']).to_csv('corpus_freq_snowball.csv')


def stop_words_gene():
    # 没有用
    words = set([])
    with open('stop_words_base.txt', 'r') as f:
        line = f.readline()
        while not line == '':
            words.add(line.strip())
            line = f.readline()
    base_count = len(words)
    '''
    corpus_snowball = pd.read_csv('corpus_freq_snowball_raw.csv', keep_default_na=False)
    for i, group_freq in enumerate(corpus_snowball['group_freq']):
        if group_freq >= 19:
            words.add(corpus_snowball['word'][i])
    
    '''
    pd.DataFrame(words, columns=['stop_word']).to_csv('stop_word.csv')


if __name__ == '__main__':
    corpus_freq_raw_gene()
    # stop_words_gene()
    corpus_freq_gene()
    s = 'don\'t he\'s a good\'d guy. Cat\'s hand is big \'s we\'ll be back\'vo'
    print(get_valid_words(s))
