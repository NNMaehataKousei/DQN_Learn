import sys, os
sys.path.append(os.pardir)
import numpy as np
from numpy.core.numeric import cross
from dqn_lesson1.gradient_2d import numerical_gradient


#t = [0,0,1,0,0,0,0,0,0,0]
#y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

#yはニューラルネットワークの出力 tは教師データ

def mean_squared_error(y,t):#二乗和誤差
    return 0.5 * np.sum((y-t)**2)

#print(mean_squared_error(np.array(y),np.array(t)))

def cross_entropy_error(y,t):#交差エントロピー誤差
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

#print(cross_entropy_error(np.array(y),np.array(t)))

def minibatch_cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

#ソフトマックス関数
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#ニューラルネットワークの学習
#勾配を求めるクラス
class simpleNet:#2x3の重みパラメータを一つだけインスタンス変数
    def __init__(self):
        self.W = np.random.randn(2,3) #ガウス分布で初期化
    
    def predict(self, x):#予測するためのメソッド
        return np.dot(x,self.W)

    def loss(self, x, t):#損失関数を求めるためのメソッド
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
#print("重みパラメータ")
#print(net.W)#重みパラメータ

x = np.array([0.6,0.9])
p = net.predict(x)
#print("最大値インデックス")
#print(p)#最大値のインデックス

t = np.array([0,0,1])#正解ラベル
#print("正解ラベル")
#print(net.loss(x,t))

#勾配を求めてみる
def f(W):
    return net.loss(x,t)

dW = numerical_gradient(f,net.W)
#print("勾配して求めた結果")
#print(dW)

f = lambda w: net.loss(x,t)#def f(w)を用いらずによい記法
dW = numerical_gradient(f, net.W)
#print(dW)