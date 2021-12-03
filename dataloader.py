import os


def loader_1():
    dir_path = r'dataset\20_newsgroups\alt.atheism'
    names = os.listdir(dir_path)
    f_path = dir_path + '\\' + names[0]
    with open(f_path, 'r') as f:
        for line in f.readlines():
            print(line)


def test():
    A = [i for i in range(10)]
    for i, ele in enumerate(A):
        if i == 3 or i == 5:
            A.remove(i)

    print(A)


if __name__ == '__main__':
    test()
