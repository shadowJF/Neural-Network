from connector import *
from activator import *

# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = np.array(sample)
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
            #print("layer output:")
            #print(output)
        return output
    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                if d % 1000 == 0:
                    print("处理到第%d个样本：" % (d))
                self.train_one_sample(labels[d], 
                    data_set[d], rate)
    def train_one_sample(self, label, sample, rate):
        #print("one sample predict:")
        #print("label:")
        #print(label)
        #print("data:")
        #print(sample)
        self.predict(sample)
        #print("calc_gradient:")
        self.calc_gradient(label)
        #print("update_weight")
        self.update_weight(rate)
    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta
    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
