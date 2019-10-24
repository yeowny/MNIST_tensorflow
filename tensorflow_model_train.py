# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowy
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, sys


def input_generator(x_data, y_data, batch_size, boolen_data_shuffle=False):
    if boolen_data_shuffle:
        train_random_seed = int(np.random.random() * 10 ** 4)
        for x in [x_data, y_data]:
            np.random.seed(train_random_seed)
            np.random.shuffle(x)

    tmp_x, tmp_y = [], []
    for i in range(len(x_data)):
        tmp_x.append(x_data[i])
        tmp_y.append(y_data[i])
        if (i + 1) % batch_size == 0 or i == len(x_data) - 1:
            yield tmp_x, tmp_y
            tmp_x, tmp_y = [], []

def model_train(Model, tf_model_path, tf_model_name, x_train, y_train, x_vali, y_vali, epoch_num=1000, batch_size=512,
                train_keep_prob=0.5, learning_rate=1e-4, early_stopping_patience=10, print_term=1,
                boolen_show_proceeding_bar=True,boolen_log_file=False):
    def early_stopping_and_save_model(sess, saver, vali_loss_list, early_stopping_patience):
        boolen_save_check = False
        if len(vali_loss_list) > early_stopping_patience + 1:
            if vali_loss_list[-early_stopping_patience - 1] < min(vali_loss_list[-early_stopping_patience:]):
                return True
        try:
            if vali_loss_list[-1] == min(vali_loss_list):
                raise ValueError
        except ValueError:
            boolen_save_check = True
            saver.save(sess, os.path.join(tf_model_path, tf_model_name, '%s.ckpt' % tf_model_name))

        if boolen_show_epoch:
            if boolen_save_check:
                sys.stdout.write('*')
            sys.stdout.write('\n')
            sys.stdout.flush()
        return False

    def show_proceeding_bar(len_batch_x_train, len_x_train, proceeding_bar_var):
        proceeding_bar_var[0] += len_batch_x_train
        proceeding_bar_print = int(proceeding_bar_var[0] * 100 / len_x_train) - proceeding_bar_var[1]
        if proceeding_bar_print != 0:
            sys.stdout.write('-' * proceeding_bar_print)
            sys.stdout.flush()

            proceeding_bar_var[1] += (int(proceeding_bar_var[0] * 100 / len_x_train) - proceeding_bar_var[1])

        return proceeding_bar_var

    log_file_list = []
    len_x_train, len_x_vali = len(x_train), len(x_vali)
    train_loss_list, vali_loss_list = [], []
    train_acc_list, vali_acc_list = [], []

    start_time = time.time()
    saver = tf.train.Saver()
    print('\n%s\n%s - training....' % ('-' * 100, tf_model_name))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(os.path.join(tf_model_path, tf_model_name,
                                                          'tensorboard_log', 'train_plot'), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(tf_model_path, tf_model_name,
                                                               'tensorboard_log', 'validation_plot'), sess.graph)

        for epoch in range(epoch_num):
            boolen_show_epoch = False
            if print_term != 0:
                if (epoch + 1) % print_term == 0:
                    boolen_show_epoch = True

            proceeding_bar_var = [0, 0]
            total_loss, total_acc, vali_total_loss, vali_total_acc = 0, 0, 0, 0

            for batch_x_train, batch_y_train in input_generator(x_train, y_train, batch_size, boolen_data_shuffle=True):
                len_batch_x_train = len(batch_x_train)
                if boolen_show_proceeding_bar and boolen_show_epoch:
                    proceeding_bar_var = show_proceeding_bar(len_batch_x_train, len_x_train, proceeding_bar_var)

                train_summary, _, loss_val, acc_val = sess.run(
                    [Model.model_summary, Model.train, Model.loss, Model.acc],
                    feed_dict={Model.x_data: batch_x_train, Model.y_data: batch_y_train,
                               Model.dropout_keep_prob: train_keep_prob,
                               Model.learning_rate: learning_rate})
                train_writer.add_summary(train_summary, global_step=epoch)
                train_writer.flush()

                total_loss += loss_val * len_batch_x_train
                total_acc += acc_val * len_batch_x_train

            train_loss_list.append(total_loss / len_x_train)
            train_acc_list.append(total_acc / len_x_train)

            try:
                vali_summary, vali_loss, vali_acc = sess.run(
                    [Model.model_summary, Model.loss, Model.acc],
                    feed_dict={Model.x_data: x_vali, Model.y_data: y_vali, Model.dropout_keep_prob: 1})

                validation_writer.add_summary(vali_summary, global_step=epoch)
                validation_writer.flush()

                vali_loss_list.append(vali_loss)
                vali_acc_list.append(vali_acc)
            except:
                for batch_x_vali, batch_y_vali in input_generator(x_vali, y_vali, batch_size):
                    len_batch_x_vali = len(batch_x_vali)
                    vali_summary, vali_loss_val, vali_acc_val = sess.run(
                        [Model.model_summary, Model.loss, Model.acc],
                        feed_dict={Model.x_data: batch_x_vali, Model.y_data: batch_y_vali, Model.dropout_keep_prob: 1})

                    validation_writer.add_summary(vali_summary, global_step=epoch)
                    validation_writer.flush()

                    vali_total_loss += vali_loss_val * len_batch_x_vali
                    vali_total_acc += vali_acc_val * len_batch_x_vali

                vali_loss_list.append(vali_total_loss / len_x_vali)
                vali_acc_list.append(vali_total_acc / len_x_vali)

            tmp_running_time = time.time() - start_time
            if boolen_log_file:
                log_file_list.append([epoch + 1, train_loss_list[-1], train_acc_list[-1],
                                      vali_loss_list[-1], vali_acc_list[-1], tmp_running_time])
            if boolen_show_epoch:
                if boolen_show_proceeding_bar:
                    sys.stdout.write('\n')
                print('#%4d/%d' % (epoch + 1, epoch_num), end='  |  ')
                print('Train: loss=%.4f/acc=%.4f' % (train_loss_list[-1], train_acc_list[-1]), end='  |  ')
                print('Validtion: loss=%.4f/acc=%.4f' % (vali_loss_list[-1], vali_acc_list[-1]), end='  |  ')
                print('%sm %ss' % (str(int(tmp_running_time // 60)).zfill(2),
                                   ('%4.2f' % (tmp_running_time % 60)).zfill(5)), end='  |  ')

            if early_stopping_and_save_model(sess, saver, vali_loss_list, early_stopping_patience):
                print('\n%s\nstop epoch : %d\n%s' % ('-' * 100, epoch - early_stopping_patience + 1, '-' * 100))
                break

    if boolen_log_file:
        pd.DataFrame(log_file_list, columns=['epoch', 'train_loss', 'train_acc', 'vali_loss', 'vali_acc', 'time']) \
            .to_csv(os.path.join(tf_model_path, tf_model_name, '%s_log.csv' % tf_model_name), index=False)

        plt.clf()
        epoch_list = [i for i in range(1, epoch + 2)]
        graph_acc_list = [train_acc_list, vali_acc_list, 'r--', 'b--', 'acc']
        graph_loss_list = [train_loss_list, vali_loss_list, 'r', 'b', 'loss']
        for train_l_a_list, vali_l_a_list, trian_color, vali_color, loss_acc in [graph_acc_list, graph_loss_list]:
            plt.plot(epoch_list, train_l_a_list, trian_color, label='train_' + loss_acc)
            plt.plot(epoch_list, vali_l_a_list, vali_color, label='validation_' + loss_acc)
            plt.xlabel('epoch')
            plt.ylabel(loss_acc)
            plt.legend(loc='lower left')
            plt.title(tf_model_name)

        plt.savefig(os.path.join(tf_model_path, tf_model_name, '%s_plot.png' % tf_model_name))
        plt.show()
