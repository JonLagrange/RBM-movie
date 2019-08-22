# -*- coding: utf-8 -*-
import sys
import pickle
import numpy as np
import tensorflow as tf


class RBM(object):

    def __init__(self, num_visible, num_hidden, dict_path, model_path):
        self.num_visible = num_visible  # 可视层节点数量
        self.num_hidden = num_hidden  # 隐藏层节点数量
        self.dict_path = dict_path + '/rbm_items.dict'
        self.model_path = model_path + '/rbm.model'
        self._init_model()

    def _init_model(self):
        # 第一层的每一个节点有一个偏差	(bias)，使用vb表示；
        # 第二层的每一个节点也有一个偏差，使用hb表示；
        self.vb = tf.placeholder("float", [self.num_visible])
        self.hb = tf.placeholder("float", [self.num_hidden])
        # 定义可视层和隐藏层之间的权重，行表示输入节点，列表示输出节点，这里权重W是一个self.num_visible * self.num_hidden的矩阵
        self.W = tf.placeholder("float", [self.num_visible, self.num_hidden])
        # 前向：改变的是隐藏层的值。输入数据经过输入层的所有节点传递到隐藏层
        self.X = tf.placeholder("float", [None, self.num_visible])
        # 从文件中读取数据
        f = open(self.dict_path, 'rb')
        self.items_dict = pickle.load(f)
        f.close()

    def _getHidden(self):
        '''
        根据输入层得到隐藏层
        self.X是一个matrix，每行代表一个样本
        '''
        # 前向：改变的是隐藏层的值。输入数据经过输入层的所有节点传递到隐藏层
        _h0 = tf.nn.sigmoid(tf.matmul(self.X, self.W) + self.hb)  # probabilities of the hidden units
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # sample_h_given_X
        return h0

    def _getVisible(self, hidden_data):
        '''
        根据隐藏层得到输入层
        hidden_data是一个matrix，每行代表一个样本
        '''
        # 反向(重构)：RBM在可视层和隐藏层之间通过多次前向后向传播重构数据
        _v1 = tf.nn.sigmoid(tf.matmul(hidden_data, tf.transpose(self.W)) + self.vb)
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))  # sample_v_given_h
        return v1, _v1

    def _load(self):
        """
        Load model params.
        """
        f = open(self.model_path, 'rb')
        model = pickle.load(f)
        f.close()
        return model

    def dataInput(self):
        """
        Get corpus and initialize model params.
        """
        visible_data = []
        for user_id, item_dict in self.items_dict.items():
            item_dicts = sorted(item_dict.items(), key=lambda x: x[0])
            item_states = []
            for item_id, item_state in item_dicts:
                item_states.append(item_state)
            visible_data.append(item_states)
        trX = np.array(visible_data)
        return trX

    def predict(self, trX, user_id, top_n):
        model = self._load()
        h0 = self._getHidden()
        v1 = self._getVisible(h0)[0]
        _v1 = self._getVisible(h0)[1]

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        user = trX[user_id-1:user_id]
        user_list = user.tolist()[0]
        pred, prob = sess.run([v1, _v1], feed_dict={self.X: user, self.W: model[0], self.vb: model[1], self.hb: model[2]})
        pred_list = pred.tolist()[0]
        prob_list = prob.tolist()[0]
        result = {}
        for i in range(len(pred_list)):
            if pred_list[i] == 1.0 and user_list[i] != 1:
                result[i + 1] = prob_list[i]
        res = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return res[:top_n]


if __name__ == '__main__':
    argv = sys.argv[1:]

    if len(argv) < 4:
        print("Error: not enough argument supplied:")
        print("predict.py <dict path> <model path> <log path> <user id>")
        exit(0)
    else:
        num_visible = 3706
        num_hidden = 500

        dict_path = argv[0]
        model_path = argv[1]
        log_path = argv[2]
        user_id = argv[3]
        rbm = RBM(num_visible, num_hidden, dict_path, model_path)
        trX = rbm.dataInput()
        movies = rbm.predict(trX, int(user_id), top_n=10)
        log = open(log_path + '/result.log', 'w')
        for movie in movies:
            log.write(str(movie) + '\n')
            print(movie)
