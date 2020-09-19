import numpy as np
from Dataprocess import load_extra_data, DNN_get_data
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#########hyper parameters############
learning_rate = 0.02
Frame_number = 25
Feature_number = 13
commitynumber = 16
batch_size = 32  # len(tags)
EPOCHTIMES = 600  # 600
#################################
networks = []
accuracies = []
cms = []


class Nerual_Networks():
    def __init__(self, frame_number=Frame_number,
                 feature_number=Feature_number,
                 time_input=1,
                 name="adam",
                 network_type="mse",
                 trainable=True,
                 lr=learning_rate,
                 model_file=None):
        self.n_features = 10
        self.network_type = network_type
        self.times_input = 1
        self.learning_rate = lr
        self.name = name
        # -------------- Network --------------
        with tf.variable_scope(self.name):
            # ------------------------- MLP -------------------------
            # MLP网络
            if 1:
                self.obs_action = tf.placeholder(tf.float32, shape=[None, feature_number * frame_number])
                self.iftraining = tf.placeholder(tf.bool, None)
                self.f1 = tf.layers.dense(inputs=self.obs_action, units=8, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                          bias_initializer=tf.constant_initializer(0.1),
                                          trainable=trainable)
                self.f3 = tf.layers.dense(inputs=self.f1, units=8, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                          bias_initializer=tf.constant_initializer(0.1),
                                          trainable=trainable)

                self.predict = tf.layers.dense(inputs=self.f3, units=self.n_features, trainable=trainable)

        # -------------- Label --------------
        if network_type == 'mse':
            self.correct_labels = tf.placeholder(tf.int32, [None])
            self.labels = tf.one_hot(self.correct_labels, 10)
            # 定义loss function
            self.square = tf.square(self.predict - self.labels)
            self.cross_entropy_mean = tf.reduce_mean(self.square, name='mse')
            # -------------- Train --------------
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy_mean)
        elif network_type == 'cross_entropy':
            self.correct_labels = tf.placeholder(tf.int32, [None])
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict,
                                                                                labels=self.correct_labels)
            # 定义loss function
            self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy, name='cross_entropy')
            # -------------- Train --------------
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy_mean)

        # -------------- Sess --------------
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # -------------- Saver --------------
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
        # Out put Information
        print("================ Build Neural Network ================")

    # Learn the model
    def learn(self, batch_features, batch_labels):
        batch_labels = batch_labels.reshape([-1])
        _, loss = self.sess.run([self.train_op, self.cross_entropy_mean], feed_dict={self.obs_action: batch_features,
                                                                                     self.correct_labels: batch_labels,
                                                                                     self.iftraining: True})
        return loss

    def test(self, correct_label, data):
        batch_predictions = tf.cast(tf.argmax(self.predict, 1), tf.int32)
        predicted_correctly = tf.equal(batch_predictions, correct_label)
        accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
        return self.sess.run(accuracy, feed_dict={self.obs_action: data, self.iftraining: False})

    def pred(self, data):
        batch_predictions = tf.cast(tf.argmax(self.predict, 1), tf.int32)
        return self.sess.run(batch_predictions, feed_dict={self.obs_action: data, self.iftraining: False})

    # 定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)



if __name__ == '__main__':
    # 解封装环境
    optimizer_type = ['mse', 'cross_entropy']
    losses = [[], []]
    accus = [[], []]
    for i in range(2):
        nn = Nerual_Networks(name=str(i), network_type=optimizer_type[i])
        # 生成新的网络对象
        tags, data = DNN_get_data('SC_MFCC.mat')
        # 导入数据
        data, testdata, tags, testtags = train_test_split(data, tags, test_size=0.2, random_state=1)
        # 分成训练集和数据集
        Epochtimes = EPOCHTIMES
        batchsize = len(tags)
        temp_accuracy = 0
        for k in range(Epochtimes):
            loss = 0
            perm = np.random.permutation(len(tags))
            for j in range(len(tags) // batchsize):
                start = j * batchsize
                end = (j + 1) * batchsize
                loss += nn.learn(batch_features=copy.deepcopy(data[start:end].reshape([-1, 25 * 13])),
                                 batch_labels=copy.deepcopy(tags[start:end].reshape([-1, 1])))
            if (k + 1) % 100 == 0:
                print('total loss', loss / len(tags))
                losses[i].append(loss / len(tags))
            if (k + 1) % 100 == 0:
                accuracy = nn.test(correct_label=testtags, data=testdata)
                # temp_accuracy = accuracy
                print('accuracy is', round(accuracy, 3))
                accus[i].append(round(accuracy, 3))
        preds = nn.pred(data=testdata)

        cm = confusion_matrix(testtags, preds)
        print(cm)
        nn.save_model('./models')
        networks.append(nn)

    plt.plot(losses[0], label='mse')
    plt.plot(losses[1], label='cross entropy+softmax')
    #  plt.plot(losses[2], label='MOMENTUM')

    plt.xlabel('epoches', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.legend()
    plt.savefig('./experiments images/mse_crossentropy_compare_loss.png')

    plt.clf()
    plt.plot(accus[0], label='mse')
    plt.plot(accus[1], label='cross entropy+softmax')

    plt.xlabel('epoches', fontsize=25)
    plt.ylabel('accuracies', fontsize=16)
    plt.legend()
    plt.savefig('./experiments images/mse_crossentropy_compare_accuracy.png')
    plt.clf()










