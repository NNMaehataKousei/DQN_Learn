import sys, os
sys.path.append(os.pardir)
from dqn_lesson1.bibun import *#functionなどを呼び込む
from dqn_lesson1.gradient_2d import numerical_gradient#勾配などの計算のメソッドを呼び込む
from dqn_lesson1.nn_l4 import cross_entropy_error
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)#オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class TwoLayerNet:
    def __init__ (self, input_size, hidden_size, output_size,weight_init_std=0.01):#引数は左から,入力層のニューロンの数、隠れ層のニューロンの数,出力層のニューロンの数。

        #重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        #params　ニュウーらるネットワークのパラメータを保持するディクショナリ変数(インスタンス変数)

    def predict(self,x):#認識(推論)を行う　引数のxは画像データ
        W1, W2 = self.params["W1"],self.params["W2"]
        b1, b2 = self.params["b1"],self.params["b2"]

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    #x入力データy教師データ
    def loss(self,x,t):#損失関数の値を求める。引数のxは画像データ,ｔは正解ラベル
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):#認識精度を求める。引数のxは画像データ,ｔは正解ラベル
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    #x入力データ,t教師データ
    def numerical_gradient(self,x,t):#重みパラメータに対する勾配を求める
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W,self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W,self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W,self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W,self.params["b2"])
        #勾配を保持するディクショナリ変数(numericalメソッドの返り値)
        #["W1"]は1層目の重みの勾配["b1"]は1層目のバイアスの勾配
        #["W1"]は2層目の重みの勾配["b1"]は2層目のバイアスの勾配

        return grads

    #高速版の勾配計算
    def gradient(self, x, t):
        W1, W2 = self.params["W1"],self.params["W2"]
        b1, b2 = self.params["b1"],self.params["b2"]
        grads = {}

        batch_num = x.shape[0]

        #forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads
