import numpy as np
from collections import OrderedDict
from common.layers import *

class NN:

    def __init__(self, input_size, hidden_size, hidden_layer_num, output_size, weight_decay_lambda, dropout_ratio):
        self.params = {}
        self.hidden_layer_num = hidden_layer_num
        self.weight_decay_lambda = weight_decay_lambda
        self.dropout_ratio = dropout_ratio
        
        # [入力サイズ, ノード数, ノード数, .... , 出力サイズ]
        all_size_list = [input_size] + [hidden_size]*hidden_layer_num + [output_size]

        for idx in range(1, len(all_size_list)):
            #  初期の重みとバイアス
            self.params['W' + str(idx)] =  np.sqrt(2.0 / all_size_list[idx - 1]) * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        self.layers = OrderedDict()
        for idx in range(1, hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            #  バッチ正規化のパラメーター
            self.params['gamma' + str(idx)] = np.ones(hidden_size)
            self.params['beta' + str(idx)] = np.zeros(hidden_size)
            self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Relu' + str(idx)] = ReLU()
            
            # drop out
            self.layers['Dropout' + str(idx)] = Dropout(self.dropout_ratio)
            
        idx = hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.lastLayer = SoftmaxWithLoss() # 出力層
        
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)      
        return x
        
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        lmd = self.weight_decay_lambda        
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            
            weight_decay += 0.5 * lmd * np.sum(W**2)

        return self.lastLayer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1 : 
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.lastLayer.backward(dout=1) 
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        lmd = self.weight_decay_lambda

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW  + lmd * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads