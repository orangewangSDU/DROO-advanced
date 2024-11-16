#  #################################################################
#  This file contains the main DROO operations, including building DNN,
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- January 2020. Written based on Tensorflow 2 by Weijian Pan and
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net  # the size of the DNN
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = keras.Sequential([
                    layers.Dense(self.net[1], activation='relu'),  # the first hidden layer
                    layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
                    # layers.Dense(self.net[3], activation='relu'),  # the third hidden layer
                    layers.Dense(self.net[-1], activation='sigmoid')  # the output layer
                ])

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = batch_memory[:, 0: self.net[0]]
        m_train = batch_memory[:, self.net[0]:]

        # print(h_train)          # (128, 10)
        # print(m_train)          # (128, 10)

        # train the DNN
        hist = self.model.fit(h_train, m_train, verbose=0)
        self.cost = hist.history['loss'][0]
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into tf placeholder
        h = h[np.newaxis, :]

        m_pred = self.model.predict(h)

        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k = 1):
        # 论文提出保持顺序的量化
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))

        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        # KNN方法量化
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


class MemoryRNN:
    def __init__(
            self,
            net,
            learning_rate=0.01,
            training_interval=10,
            batch_size=100,
            memory_size=1000,
            output_graph=False
    ):
        self.net = net
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # Create RNN model with multiple layers and dropout
        self._build_net()

    def _build_net(self):
        # 使用了 LSTM 层来构建模型
        model = keras.Sequential()
        model.add(layers.Input(shape=(1, self.net[0])))  # 使用 Input 层指定输入形状
        model.add(layers.LSTM(self.net[1], return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(self.net[1], return_sequences=False))
        model.add(layers.Dense(self.net[-1], activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.losses.binary_crossentropy,
                      metrics=['accuracy'])

        self.model = model

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        h_train = batch_memory[:, 0:self.net[0]]

        # 修改这一行，将 m_train 的形状调整为 (批量大小, 输出维度)
        m_train = batch_memory[:, self.net[0]:]  # 取出输出并保持原始形状

        # 现在 h_train 的形状为 (批量大小, 特征数量)
        hist = self.model.fit(h_train.reshape(-1, 1, self.net[0]), m_train, verbose=0)  # 适当调整输入形状
        self.cost = hist.history['loss'][0]
        self.cost_his.append(self.cost)

    def decode(self, h, k=1, mode='OP'):
        h = h[np.newaxis, np.newaxis, :]  # 扩展维度以适应 RNN 输入 (1, 1, 10)
        m_pred = self.model.predict(h)

        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k=1):
        m_list = []
        m_list.append(1 * (m > 0.5))

        if k > 1:
            m_abs = abs(m - 0.5)
            idx_list = np.argsort(m_abs)[:k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k=1):
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


class MemoryCNN:
    def __init__(
        self,
        net,
        learning_rate=0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):
        self.net = net  # the size of the CNN network
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Store all binary actions
        self.enumerate_actions = []
        self.memory_counter = 0  # Initialize memory counter at 0
        self.cost_his = []  # Store training cost history
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # Construct CNN network
        self._build_net()

    def _build_net(self):
        self.model = keras.Sequential([
            layers.Input(shape=(self.net[0], 1)),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(self.net[1], activation='relu'),
            layers.Dropout(0.5),  # Adding Dropout layer
            layers.Dense(self.net[-1], activation='sigmoid')
        ])

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                           loss=tf.losses.binary_crossentropy,
                           metrics=['accuracy'])

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter >= self.memory_size and self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter >= self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        h_train = batch_memory[:, 0:self.net[0]]
        m_train = batch_memory[:, self.net[0]:]

        # Reshape h_train to (batch_size, features, 1)
        h_train = h_train.reshape(-1, self.net[0], 1)  # Correct shape (batch_size, features, 1)

        # Train the CNN
        hist = self.model.fit(h_train, m_train, verbose=0)
        self.cost = hist.history['loss'][0]
        assert (self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k=1, mode='OP'):
        h = h[np.newaxis, :, np.newaxis]  # Expand dimensions to (1, features, 1)
        m_pred = self.model.predict(h)

        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k=1):
        m_list = []
        m_list.append(1 * (m > 0.5))
        if k > 1:
            m_abs = abs(m - 0.5)
            idx_list = np.argsort(m_abs)[:k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0.5:
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))
        return m_list

    def knn(self, m, k=1):
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        sqd = ((self.enumerate_actions - m) ** 2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()