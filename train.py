# -*- coding: utf-8 -*-
import os
import sys
import pickle
import numpy as np
import tensorflow as tf


class RBM(object):

    def __init__(self, num_visible, num_hidden, lr, epochs, bach_size, dict_path, model_path):
        self.num_visible = num_visible  # 可视层节点数量
        self.num_hidden = num_hidden  # 隐藏层节点数量
        self.lr = lr  # 学习率
        self.epochs = epochs  # 训练周期
        self.batch_size = bach_size  # batch大小
        self.dict_path = dict_path + '/rbm_items.dict'  # 输入数据路径
        self.model_path = model_path + '/rbm.model'  # 模型保存路径
        self.gpu_index = [1, 2, 3, 4]  # 设置GPU
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
        # 从文件中读取输入数据
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

    def _dataInit(self):
        self.cur_w = np.zeros([self.num_visible, self.num_hidden], np.float32)
        self.cur_vb = np.zeros([self.num_visible], np.float32)
        self.cur_hb = np.zeros([self.num_hidden], np.float32)
        self.prv_w = np.zeros([self.num_visible, self.num_hidden], np.float32)
        self.prv_vb = np.zeros([self.num_visible], np.float32)
        self.prv_hb = np.zeros([self.num_hidden], np.float32)

    def _save(self, model):
        """
        Save model params.
        """
        f = open(self.model_path, 'wb')
        pickle.dump(model, f)
        f.close()

    def _set_gpus(self):
        gpus = '0'
        if type(self.gpu_index) == list:
            gpus = ','.join(str(_) for _ in self.gpu_index)
        if type(self.gpu_index) == int:
            gpus = str(self.gpu_index)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

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

    def train(self, trX):
        '''
        对比散度(Contrastive Divergence (CD))，它是按照这样的方式去
        设计的：梯度估计的方向至少有一些准确。对比散度实际是一个用来
        计算和调整权重矩阵的一个矩阵。改变权重W渐渐地变成了权重值的训练。
        然后在每一步(epoch)，W通过下面的公式被更新为一个新的值W′。
                            W′=W+α∗CD
        α是很小的步长，也就是大家所熟悉的学习率(Learning rate)。
        '''
        self._set_gpus()
        h0 = self._getHidden()
        v1 = self._getVisible(h0)[0]
        h1 = tf.nn.sigmoid(tf.matmul(v1, self.W) + self.hb)

        w_pos_grad = tf.matmul(tf.transpose(self.X), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)
        CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(self.X)[0])
        update_w = self.W + self.lr * CD
        update_vb = self.vb + self.lr * tf.reduce_mean(self.X - v1, 0)
        update_hb = self.hb + self.lr * tf.reduce_mean(h0 - h1, 0)

        # 计算误差：每一步(epoch)， 我们计算从第1步到第n步的平方误差的和，这显示了数据和重构数据的误差
        loss = tf.reduce_mean(tf.square(self.X - v1))

        self._dataInit()
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        model = []
        errors = []
        for epoch in range(self.epochs):
            for start, end in zip(range(0, len(trX), self.batch_size), range(self.batch_size, len(trX), self.batch_size)):
                batch = trX[start:end]
                self.cur_w = sess.run(update_w, feed_dict={self.X: batch, self.W: self.prv_w, self.vb: self.prv_vb, self.hb: self.prv_hb})
                self.cur_vb = sess.run(update_vb, feed_dict={self.X: batch, self.W: self.prv_w, self.vb: self.prv_vb, self.hb: self.prv_hb})
                self.cur_hb = sess.run(update_hb, feed_dict={self.X: batch, self.W: self.prv_w, self.vb: self.prv_vb, self.hb: self.prv_hb})
                self.prv_w = self.cur_w
                self.prv_vb = self.cur_vb
                self.prv_hb = self.cur_hb
                if start % 100 == 0:
                    errors.append(sess.run(loss, feed_dict={self.X: trX, self.W: self.cur_w, self.vb: self.cur_vb, self.hb: self.cur_hb}))
                    model.append([self.cur_w, self.cur_vb, self.cur_hb])
            print('Epoch: %d' % (epoch + 1), 'Reconstruction loss: %f' % errors[-1])

        uw = model[-1]
        self._save(uw)


if __name__ == '__main__':
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path>")
        exit(0)
    else:
        num_visible = 3706
        num_hidden = 500
        lr = 1.0
        epochs = 10
        bach_size = 128
        input_path = argv[0]
        output_path = argv[1]
        rbm = RBM(num_visible, num_hidden, lr, epochs, bach_size, input_path, output_path)
        trX = rbm.dataInput()
        rbm.train(trX)
