from gzip import _GzipReader
import numpy as np

#乗算レイヤ
class MulLayer:
    #順伝播時の入力値を保持するために用いる
    def __init__(self):
        self.x = None
        self.y = None
    
    #x,yの二つの引数を受け取り、乗算して出力する
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    #上流から伝わってきた微分に対して、逆伝播のひっくり返した値を乗算して下流に流す
    def backward(self, dout):
        dx = dout * self.y #xとyをひっくり返す
        dy = dout * self.x

        return dx, dy

""""
###############################################################
#コードサンプル例
apple = 100
apple_num = 2
tax = 1.1

##Layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

##forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

##backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_tax_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
###########################################################
"""

#加算レイヤ
class AddLayer:
    def __init__(self):#加算レイヤでは初期化は必要ない
        pass

    def forward(self, x, y):#xとyを加算して出力
        out = x + y
        return out

    def backward(self, dout):#上流から伝わってきた微分をそのまま下流に流すだけ
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
##############################################################
#サンプルコード　逆伝播.jpgを実際にコードとして実行
"""
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_Layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_Layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_Layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_num, dapple, dorange, dorange_num, dtax)

"""
#########################################################################

#ニューラルネットワークのためのレイヤを作成
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):#引数にはNumpyの配列を入力
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):#引数にはNumpyの配列を入力
        dout[self.mask] = 0
        dx  = dout

        return dx

#Relu動作確認
"""
x = np.array([[1.0,-0.5], [-2.0, 3.0]])
print(x)
mask = (x <=0 )
print(mask)
"""
class Sigmoid():
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

#Affine / SOftmaxレイヤの実装
#重要：ニューラルネットワークでの行列の乗算ではそれぞれ対応しなければならない
#例 行列の形(2.) x 行列の形(2, 3) = 行列の形(3.)　2と2、3と3のように。そのため逆伝播時にはそれに対応するために転置を使い行列を変換する必要がある

x_dot_W = np.array([[0,0,0], [10,10,10]])
B = np.array([1, 2, 3])
print(x_dot_W)

x_dot_W = x_dot_W + B
print(x_dot_W)

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

#Softmax-with-Lossレイヤ
#ソフトマックス関数とは入力された値を正規化して出力することである。
#逆伝播する際に伝播する値をバッチの個数(batch_size)で割ることでデータ1個あたりの誤差が前レイヤへ伝播する点に注意すること
def softmax(x):
    #x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    #return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None#損失
        self.y = None#ソフトマックの出力
        self.t = None#教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx

#凡庸関数の実装
def numerical_gradient(f, x,):#偏微分
    h = 1e-4
    grad = np.zeros_like(x)#xと同じ形状の配列を生成

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad



