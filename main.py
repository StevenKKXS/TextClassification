import numpy as np
from tqdm import tqdm
from naive_bayes import NaiveBayes
from nltk.stem import SnowballStemmer
from k_nn import K_NN


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # my_k_nn = K_NN(para_k=6)
    # my_k_nn.gene_simple_train_set()
    # my_k_nn.pre_process()
    # my_k_nn.train()
    # my_k_nn.predict()
    # my_k_nn.k_cross_validation(k=5)
    nb = NaiveBayes('dataset', 'stop_words_base.txt')
    nb.k_cross_validation(k=5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
