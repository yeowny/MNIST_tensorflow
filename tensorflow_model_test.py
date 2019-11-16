# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowny
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os


def model_test(tf_model_path, tf_model_name, x_test, y_test, batch_size=512,
               tf_model_important_var_name='important_vars_ops'):
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph(os.path.join(tf_model_path, tf_model_name, '%s.ckpt.meta' % tf_model_name))
        saver.restore(sess, os.path.join(tf_model_path, tf_model_name, '%s.ckpt' % tf_model_name))
        x_data, y_data, dropout_keep_prob, _, loss, acc, pred_y, _ = tf.get_collection(tf_model_important_var_name)

        try:
            test_loss, test_acc, test_pred_y = sess.run(
                [loss, acc, pred_y], feed_dict={x_data: x_test, y_data: y_test, dropout_keep_prob: 1})
        except:
            from tensorflow_model_train import input_generator

            len_x_test = len(x_test)
            test_total_loss, test_total_acc = 0, 0
            test_pred_y = []
            for batch_x_test, batch_y_test in input_generator(x_test, y_test, batch_size):
                len_batch_x_test = len(batch_x_test)
                test_loss_val, test_acc_val, test_pred_y_val = sess.run(
                    [loss, acc, pred_y], feed_dict={x_data: batch_x_test, y_data: batch_y_test, dropout_keep_prob: 1})
                test_pred_y.extend(test_pred_y_val)
                test_total_loss += test_loss_val * len_batch_x_test
                test_total_acc += test_acc_val * len_batch_x_test

            test_loss = test_total_loss / len_x_test
            test_acc = test_total_acc / len_x_test

    test_true_y = np.argmax(y_test, axis=1)
    print('\n' + '=' * 100)
    print(classification_report(test_true_y, test_pred_y, target_names=['mnist_' + str(i) for i in range(10)]))
    print(pd.crosstab(pd.Series(test_true_y), pd.Series(test_pred_y), rownames=['True'], colnames=['Predicted'],
                      margins=True))
    print('\n%s\n%s\n' % ('=' * 100, tf_model_name))
    print('Test_loss = %.4f\nTest_acc = %.4f\n%s' % (test_loss, test_acc, '=' * 100))
