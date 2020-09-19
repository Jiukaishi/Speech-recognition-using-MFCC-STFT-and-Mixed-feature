import numpy as np
from Dataprocess import get_data,get_stft_data
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#########hyper parameters############
learning_rate = 0.025
Frame_number = 25
Feature_number = 13
class Nerual_Networks():
    def __init__(self,frame_number=Frame_number,
                 feature_number=Feature_number,
                 time_input=1,
                 network_type="CNN",
                 trainable=True,
                 lr = learning_rate,
                 model_file=None):
        self.n_features = 10
        self.network_type = network_type
        # if self.network_type == "CNN":
        #     self.times_input = time_input
        if self.network_type == "Basic_RNN":
            self.times_input = time_input
        else:
            self.times_input = 1
        self.learning_rate = lr
        self.batch = 32 #SGD
        self.hiddennum = 8
        self.name = str(copy.deepcopy(network_type))
        # -------------- Network --------------
        with tf.variable_scope(self.name):
            # ------------------------- MLP -------------------------
            # MLP网络
            if network_type == "ANN":
                self.obs_action = tf.placeholder(tf.float32, shape=[None, feature_number*frame_number])
                self.f1 = tf.layers.dense(inputs=self.obs_action, units=64, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                          bias_initializer=tf.constant_initializer(0.1),
                                          trainable=trainable)
                self.f2 = tf.layers.dense(inputs=self.f1, units=32, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                          bias_initializer=tf.constant_initializer(0.1),
                                          trainable=trainable)
                self.f3 = tf.layers.dense(inputs=self.f2, units=16, activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                          bias_initializer=tf.constant_initializer(0.1),
                                          trainable=trainable)
                self.predict = tf.layers.densze(inputs=self.f3, units=self.n_features, trainable=trainable)
            elif network_type == "CNN":
                self.obs_action = tf.placeholder(tf.float32, shape=[None, frame_number, feature_number, 1])
                conv1 = tf.layers.conv2d(
                    inputs=self.obs_action,
                    filters=32,
                    kernel_size=[4, 4],
                    strides=1,
                    padding='same',
                    activation=tf.nn.relu
                )
                print(conv1)

        
                pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=[2, 2],
                    strides=2
                )
                print(pool1)
   
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[4, 4],
                    strides=1,
                    padding='same',
                    activation=tf.nn.relu
                )

                pool2 = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=[2, 2],
                    strides=2
                )

                flat = tf.reshape(pool2, [-1, 6 * 3 * 64])


                dense = tf.layers.dense(
                    inputs=flat,
                    units=1024,
                    activation=tf.nn.relu
                )
                # training 是否是在训练的时候丢弃
                dropout = tf.layers.dropout(
                    inputs=dense,
                    rate=0.5,
                )
                print(dropout)
                self.predict= tf.layers.dense(
                    inputs=dropout,
                    units=10
                )
        # -------------- Label --------------
        self.correct_labels  = tf.placeholder(tf.int32, [None])
        # targets = tf.squeeze(tf.cast(targets, tf.int32))
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict, labels=self.correct_labels)
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
        if self.network_type == "CNN":
            print("---------------- Network Type: CNN ----------------")
        else:
            if self.network_type == "MLP":
                print("---------------- Network Type: Build MLP ----------------")
            else:
                print("---------------- Network Type: Basic RNN ----------------")
    # Learn the model
    def learn(self, batch_features, batch_labels):
        batch_labels = batch_labels.reshape([-1])
        _, loss = self.sess.run([self.train_op, self.cross_entropy_mean], feed_dict={self.obs_action: batch_features,
                                                                       self.correct_labels: batch_labels})
        return loss

    def test(self, correct_label, data):
        batch_predictions = tf.cast(tf.argmax(self.predict, 1), tf.int32)
        predicted_correctly = tf.equal(batch_predictions, correct_label)
        accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
        return self.sess.run(accuracy,feed_dict={self.obs_action: data})
    def pred(self, data):
        batch_predictions = tf.cast(tf.argmax(self.predict, 1), tf.int32)
        return self.sess.run(batch_predictions, feed_dict={self.obs_action: data})
    # 定义存储模型函数
    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    # 定义恢复模型函数
    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

if __name__ == '__main__':                                                                                              # 解封装环境
    nn = Nerual_Networks()
    tags, data = get_data('SC_STFT.mat')
    # 导入数据
    data, testdata, tags, testtags = train_test_split(data, tags, test_size=0.2, random_state=1)
    testdata = testdata.reshape([-1,25,13 ,1])
    Epochtimes = 600
    losses = []
    accuracies = []
    batchsize=len(data)
    for i in range(Epochtimes):
        loss = 0
        perm = np.random.permutation(len(tags))
        for j in range(len(tags)//batchsize):
            loss += nn.learn(batch_features=copy.deepcopy(data[j*batchsize :(j+1)*batchsize].reshape([-1, 25, 13, 1])), batch_labels=copy.deepcopy(tags[j*batchsize :(j+1)*batchsize].reshape([-1, 1])))
        losses.append(loss/len(tags))
        if (i + 1) % 10 == 0:
            print('total loss', loss/len(tags))
        if (i+1)%100==0:

            accuracy = nn.test(correct_label=testtags, data=testdata)
            accuracies.append(accuracy)
            print('accuracy is', accuracy)
            plt.plot(losses)
            plt.savefig('loss_!.png')
            plt.clf()
            plt.plot(accuracies)
            plt.savefig('accuracy_1.png')
            plt.clf()
            print(losses)
            print(accuracies)
    preds = nn.pred(data=testdata)

    cm = confusion_matrix(testtags, preds)
    print(cm)








