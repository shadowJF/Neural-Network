import numpy as np

# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        l = []
        for num in weighted_input:
            if num <= -700:
                value = 0
            else:
                value = 1.0 / (1.0 + np.exp(-num))
            l.append(value)
        return np.array(l)

    def backward(self, output):
        return output * (1 - output)
