# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowny
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowy
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os, pickle


def mnist_load_data(mnist_file_path='', mnist_pkl_name='', bool_save_pkl=False):
    try:
        with open(os.path.join(mnist_file_path, mnist_pkl_name), 'rb') as f:
            mnist_x_train, mnist_y_train, mnist_x_vali, mnist_y_vali, mnist_x_test, mnist_y_test = pickle.load(f)
            print('load mnist pickle file')
    except FileNotFoundError:
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets(mnist_file_path, one_hot=True)
        print('load mnist data')

        mnist_x_train, mnist_y_train = mnist.train.images, mnist.train.labels
        mnist_x_vali, mnist_y_vali = mnist.validation.images, mnist.validation.labels
        mnist_x_test, mnist_y_test = mnist.test.images, mnist.test.labels
        del mnist

        if bool_save_pkl:
            print('make mnist pickle file')
            with open(os.path.join(mnist_file_path, mnist_pkl_name), 'wb') as f:
                pickle.dump([mnist_x_train, mnist_y_train, mnist_x_vali, mnist_y_vali, mnist_x_test, mnist_y_test],
                            f)

    return mnist_x_train, mnist_y_train, mnist_x_vali, mnist_y_vali, mnist_x_test, mnist_y_test

def show_mnist_data(x_data, y_data, data_index):
    ex_img = x_data[data_index]
    ex_label = y_data[data_index]

    plt.clf()
    plt.imshow(ex_img.reshape(28, 28), cmap='gray')
    plt.show()
    print(np.argmax(ex_label), ex_label)


if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError as e:
        print('{0}\n{1}\n{0}'.format('!' * 50, e))
# =============================================================================
    mnist_file_path = 'mnist_data'
    mnist_pkl_name = 'mnist_data.pkl'
# =============================================================================
    mnist_x_train, mnist_y_train, mnist_x_vali, mnist_y_vali, mnist_x_test, mnist_y_test\
        = mnist_load_data(mnist_file_path, mnist_pkl_name, bool_save_pkl=True)
    show_mnist_data(mnist_x_train, mnist_y_train, 1234)
    print('%s\ntrain set : %s / %s' % ('#'*100, mnist_x_train.shape, mnist_y_train.shape))
    print('validation set : %s / %s' % (mnist_x_vali.shape, mnist_y_vali.shape))
    print('test set : %s / %s\n%s' % (mnist_x_test.shape, mnist_y_test.shape, '#'*100))
