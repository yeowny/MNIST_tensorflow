# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowy
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import tensorflow as tf
import os


class mnist_RNN_model:
    def __init__(self):
        self.tf_model_important_var_name = 'important_vars_ops'

        self.rnn_layer_1_node = 256
        self.rnn_layer_2_node = 512
        self.fc_layer_node = 128

        self.x_data, self.y_data, self.dropout_keep_prob, self.learning_rate, self.loss, self.acc, self.pred_y, \
        self.softmax_oupt, self.train, self.model_summary = self.create_model()

    def create_model(self):
        tf.reset_default_graph()

        x_data = tf.placeholder(tf.float32, [None, 28 * 28], name='x_data')
        re_x_data = tf.reshape(x_data, [-1, 28, 28])
        y_data = tf.placeholder(tf.float32, [None, 10], name='y_data')

        dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # # ====================================================================================================
        # with tf.variable_scope('rnn_layer'):
        #     # cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_layer_1_node, activation=tf.nn.tanh)
        #     # cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_layer_1_node)
        #     cell = tf.nn.rnn_cell.GRUCell(self.rnn_layer_1_node)
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        #
        #     rnn_oupt, _ = tf.nn.dynamic_rnn(cell, re_x_data, dtype=tf.float32)
        #     result_rnn_oupt = rnn_oupt[:, -1, :]
        # # ====================================================================================================

        # # ====================================================================================================
        # with tf.variable_scope('multi_rnn_layer'):
        #     cell = tf.contrib.rnn.MultiRNNCell(
        #         [tf.nn.rnn_cell.GRUCell(self.rnn_layer_1_node), tf.nn.rnn_cell.GRUCell(self.rnn_layer_2_node)])
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        #
        #     rnn_oupt, _ = tf.nn.dynamic_rnn(cell, re_x_data, dtype=tf.float32)
        #     result_rnn_oupt = rnn_oupt[:, -1, :]
        # # ====================================================================================================

        # # ====================================================================================================
        # with tf.variable_scope('bidirectional_rnn_layer'):
        #     fw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_layer_1_node)
        #     fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        #
        #     bw_cell = tf.nn.rnn_cell.GRUCell(self.rnn_layer_1_node)
        #     bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        #
        #     rnn_oupt, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, re_x_data, dtype=tf.float32)
        #     result_rnn_oupt = tf.concat(rnn_oupt, axis=2)[:, -1, :]
        # # ====================================================================================================

        # ====================================================================================================
        with tf.variable_scope('bidirectional_muti_rnn_layer'):
            fw_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(self.rnn_layer_1_node), tf.nn.rnn_cell.GRUCell(self.rnn_layer_2_node)])
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)

            bw_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(self.rnn_layer_1_node), tf.nn.rnn_cell.GRUCell(self.rnn_layer_2_node)])
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)

            rnn_oupt, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, re_x_data, dtype=tf.float32)
            result_rnn_oupt = tf.concat(rnn_oupt, axis=2)[:, -1, :]
        # ====================================================================================================

        with tf.variable_scope('fc_layer'):
            fc_layer_z = tf.layers.dense(result_rnn_oupt, self.fc_layer_node, activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            fc_layer_d = tf.nn.dropout(fc_layer_z, dropout_keep_prob)

        with tf.variable_scope('output_layer'):
            outp_w = tf.get_variable('weight', [self.fc_layer_node, y_data.shape.as_list()[-1]],
                                     initializer=tf.random_normal_initializer(stddev=0.01))
            outp_u = tf.matmul(fc_layer_d, outp_w)

        with tf.variable_scope('Accuracy'):
            softmax_oupt = tf.nn.softmax(outp_u, name='softmax_oupt')
            pred_y = tf.argmax(softmax_oupt, 1, name='pred_y')
            acc = tf.reduce_mean(tf.cast(tf.equal(pred_y, tf.argmax(y_data, 1)), tf.float32), name='acc')

        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outp_u, labels=y_data), name='loss')

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        for op in [x_data, y_data, dropout_keep_prob, learning_rate, loss, acc, pred_y, softmax_oupt]:
            tf.add_to_collection(self.tf_model_important_var_name, op)

        tf.summary.scalar('epoch_loss', loss)
        tf.summary.scalar('epoch_acc', acc)
        model_summary = tf.summary.merge_all()

        return x_data, y_data, dropout_keep_prob, learning_rate, loss, acc, pred_y, softmax_oupt, train, model_summary


if __name__ == "__main__":
    from tensorflow_load_mnist_data import mnist_load_data, show_mnist_data
    from tensorflow_model_train import model_train
    from tensorflow_model_test import model_test

    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError as e:
        print('{0}\n{1}\n{0}'.format('!' * 50, e))
# =============================================================================
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# =============================================================================
    mnist_file_path = 'mnist_data'
    mnist_pkl_name = 'mnist_data.pkl'

    tf_model_path = 'tf_model'
    tf_model_name = 'mnist_RNN_model'
# =============================================================================
    mnist_x_train, mnist_y_train, mnist_x_vali, mnist_y_vali, mnist_x_test, mnist_y_test\
        = mnist_load_data(mnist_file_path, mnist_pkl_name, boolen_save_pkl=True)
    # show_mnist_data(mnist_x_train, mnist_y_train, 1234)
    print('%s\ntrain set : %s / %s' % ('#'*100, mnist_x_train.shape, mnist_y_train.shape))
    print('validation set : %s / %s' % (mnist_x_vali.shape, mnist_y_vali.shape))
    print('test set : %s / %s\n%s' % (mnist_x_test.shape, mnist_y_test.shape, '#'*100))

    Model = mnist_RNN_model()
    model_train(Model, tf_model_path, tf_model_name, mnist_x_train, mnist_y_train, mnist_x_vali, mnist_y_vali,
                early_stopping_patience=10, print_term=1, boolen_show_proceeding_bar=True, boolen_log_file=True)
    model_test(tf_model_path, tf_model_name, mnist_x_test, mnist_y_test)

# ====================================================================================================
# mnist_RNN_model
#
# Test_loss = 0.0446
# Test_acc = 0.9889
# ====================================================================================================
