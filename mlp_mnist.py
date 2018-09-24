# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time



def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))


class mlp(object):
    def __init__(self, lr=0.1, momentum=0.5, lda=0.0, te=1e-5, epoch=5, size=None):
        self.learningRate = lr
        self.lambda_ = lda
        self.thresholdError = te
        self.maxEpoch = epoch
        self.size = size
        self.momentum = momentum
        self.W = []
        self.b = []
        self.last_W = []
        self.last_b = []
        self.init()

    def init(self):
        for i in range(len(self.size) - 1):  # 初始化权重和偏执
            self.W.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], self.size[i]))))
            self.b.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.size[i + 1], 1))))

    def forwardPropagation(self, item=None):
        a = [item]
        for wIndex in range(len(self.W)):
            a.append(sigmod(self.W[wIndex] * a[-1] + self.b[wIndex]))  # 实现sigmod(wx+b)的各层前向传播
        return a

    def backPropagation(self, label=None, a=None):
        # print ("backPropagation--------------------begin")
        delta = []
        delta.append(
            np.multiply((a[-1] - label), np.multiply(a[-1], (1.0 - a[-1]))))  # 输出层的输入神经元 delat值（经过sigmod激活函数之前）
        for i in range(len(self.W) - 1):
            abc = np.multiply(a[-2 - i], 1 - a[-2 - i])  # 输出层sigmod激活函数的偏导数
            cba = np.multiply(self.W[-1 - i].T * delta[-1], abc)  # 隐藏层的神经元 delta值（经过sigmod激活函数之前）
            delta.append(cba)  # 此处每层神经元的delta值存放是从后往前堆放的（经过sigmod激活函数之前）

        if not len(self.last_W):  # 不存在时，表示第一次反向传播，不存在前一次的动量项
            for i in range(len(self.size) - 1):
                self.last_W.append(np.mat(np.zeros_like(self.W[i])))
                self.last_b.append(np.mat(np.zeros_like(self.b[i])))
            for j in range(len(delta)):  # 连接权重偏置的参数更新
                ads = delta[j] * a[-2 - j].T  # 对应层连接权重的 delta值
                self.W[-1 - j] = self.W[-1 - j] - self.learningRate * (
                            ads + self.lambda_ * self.W[-1 - j])  # W-lr*delta_W
                self.b[-1 - j] = self.b[-1 - j] - self.learningRate * delta[j]  # b-lr*delta_b
                self.last_W[-1 - j] = -self.learningRate * (ads + self.lambda_ * self.W[-1 - j])  # 记录此次权重参数更新方向，用于添加动量项
                self.last_b[-1 - j] = -self.learningRate * delta[j]
        else:
            for j in range(len(delta)):  # 连接权重的参数更新
                ads = delta[j] * a[-2 - j].T  # 对应层连接权重的 delta值
                self.W[-1 - j] = self.W[-1 - j] - self.learningRate * (
                            ads + self.lambda_ * self.W[-1 - j]) + self.momentum * self.last_W[-1 - j]
                self.b[-1 - j] = self.b[-1 - j] - self.learningRate * delta[j] + self.momentum * self.last_b[-1 - j]
                self.last_W[-1 - j] = -self.learningRate * (ads + self.lambda_ * self.W[-1 - j]) + self.momentum * \
                                      self.last_W[-1 - j]
                self.last_b[-1 - j] = -self.learningRate * delta[j] + self.momentum * self.last_b[-1 - j]
        error = sum(0.5 * np.multiply(a[-1] - label, a[-1] - label))  # L2 loss
        return error

    def train(self, input_=None, target=None, show=1):
        # print("train begin")
        # plt.ion()
        # plt.figure(1)
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.grid(True)
        # plt.title('epoch-loss')
        for ep in range(self.maxEpoch):
            print(ep)
            # plt.draw()
            error = []
            for itemIndex in range(input_.shape[1]):
                a = self.forwardPropagation(input_[:, itemIndex])
                e = self.backPropagation(target[:, itemIndex], a)
                # print("e:",len(e),len(e[0]))
                error.append(e[0, 0])
            epoch_error = sum(error) / len(error)

            # plt.scatter(ep, epoch_error)

            if epoch_error < self.thresholdError:
                print("Finish {0}: ".format(ep), epoch_error)
                return
            elif ep % show == 0:
                print("epoch {0}: ".format(ep), epoch_error)

    def sim(self, inp=None):
        return self.forwardPropagation(item=inp)[-1]

def one_hot(y):
    y_one_hot = np.zeros((len(y), 10))
    for i, label in enumerate(y):
        y_one_hot[i, label] = 1
    return y_one_hot


if __name__ == "__main__":

    output = 10  # 输出数据大小
    input = 784  # 输入数据大小

    import loader


    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    train_x,train_y=loader.load_mnist('./mnist/',kind='train')
    test_x,test_y=loader.load_mnist('./mnist/',kind='t10k')
    # In this example, we limit mnist data
    train_y=one_hot(train_y)
    test_y=one_hot(test_y)
    Xtr, Ytr = train_x[:50000],train_y[:50000] # 5000 for training (nn candidates)

    Xte, Yte = test_x[:5000],test_y[:5000]

    train_samples = []
    train_labels = []
    for i in range(len(Xtr)):
        train_samples.append(Xtr[i])
        train_labels.append(Ytr[i])
    train_samples = np.mat(train_samples).transpose()
    train_labels = np.mat(train_labels).transpose()
    print("train_x:",train_samples.shape,"train_y:",train_labels.shape)

    test_samples = []
    test_labels = []
    for i in range(len(Xte)):
        test_samples.append(Xte[i])
        test_labels.append(Yte[i])
    test_samples = np.mat(test_samples).transpose()
    test_labels = np.mat(test_labels).transpose()
    print("test_x:",test_samples.shape,"test_y:",test_labels.shape)

    # 根据训练数据初始化模型
    # lr 学习率
    # momentum 动量项建议（0.1-0.8）
    # lda 正则化参数
    # epoch 最大迭代次数
    # te 训练误差上限
    # 网络结构：输入数据大小size[0],输出数据size[-1]， size 可以设置网络结构 eg[input, 10, 10, 10, output]
    # model = mlp(lr=0.1, momentum=0.5, lda=0.0, te=1e-5, epoch=100, size=[input, 128, 128, output])
    model = mlp(lr=0.5, momentum=0.5, lda=0.0, te=1e-5, epoch=5, size=[input, 10, 10,10, output])

    # 这里没有batch_size更新,所有数据依次单独更新网络，也就是用的single data SGD，整个数据遍历一遍为1个epoch
    model.train(input_=train_samples, target=train_labels, show=1)

    # plt.figure(2)
    # 训练数据的预测
    # plt.subplot(211)
    sims = []
    print("show:",np.sum(model.sim(train_samples[:, 2])))
    [sims.append(model.sim(train_samples[:, idx])) for idx in range(train_samples.shape[1])]
    print( "training error: ", sum(np.array(sum(0.5 * np.multiply(train_labels - np.mat(np.array(sims).transpose()),
                                                           train_labels - np.mat(np.array(sims).transpose())) /
                                         train_labels.shape[1]).tolist()[0])))

    # plt.plot(list(range(train_labels.shape[1])), train_labels.argmax(axis=0).tolist()[0], 'b', linestyle='-.', label='gt', )
    # plt.plot(list(range(train_labels.shape[1])), np.mat(np.array(sims)).T.argmax(axis=0).tolist()[0], 'r', linestyle='--',
    #          label='predict')
    # plt.legend(loc='upper right')
    # plt.xlabel('x-labels')
    # plt.ylabel('y-labels')
    # plt.grid(True)
    # plt.title('train')

    # 测试数据的预测
    # plt.subplot(212)
    sims_test = []
    [sims_test.append(model.sim(test_samples[:, idx])) for idx in range(test_samples.shape[1])]
    print("test error: ", sum(np.array(sum(0.5 * np.multiply(test_labels - np.mat(np.array(sims_test).transpose()),
                                                       test_labels - np.mat(np.array(sims_test).transpose())) /
                                     test_labels.shape[1]).tolist()[0])))

    # plt.plot(list(range(test_labels.shape[1])), test_labels[0].tolist()[0], 'b', linestyle='-.', label='gt')
    # plt.plot(list(range(test_labels.shape[1])), np.mat(np.array(sims_test)).T[0].tolist()[0], 'r', linestyle='--',
    #          label='predict')
    # plt.legend(loc='upper right')
    # plt.xlabel('x-labels')
    # plt.ylabel('y-labels')
    # plt.grid(True)
    # plt.title('val')
    # plt.draw()
    # time.sleep(1000)
    # plt.show()