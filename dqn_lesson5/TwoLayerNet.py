"""
誤差逆伝播法の実装
まずニューラルネットワークの学習手順の確認
前提
　ニューラルネットワークは、適応可能な重みとバイアスがあり、この重みとバイアスを訓練データに適応するように調整すること(学習)

ＳＴＥＰ１　ミニバッチ
　訓練データの中からランダムに一部のデータを選び出す

ＳＴＥＰ２　勾配の算出
　各重みパラメータに関する損失関数の勾配を求める

ＳＴＥＰ３　パラメータの更新
　重みパラメータを勾配方向に微小量だけ更新する

ＳＴＥＰ４　繰り返す
　ステップ１からステップ３を繰り返す

ポイント：誤差逆伝播法を用いれば、時間を要する数値微分とは違い、高速で効率よく勾配を求めることができる！
"""
#今回は2層のニューラルネットワークを使って学習する

import sys, os
from typing import NewType
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from dqn_lesson5.MulLayer import Affine
from dqn_lesson5.MulLayer import numerical_gradient
from dqn_lesson5.MulLayer import Relu
from dqn_lesson5.MulLayer import SoftMaxWithLoss

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        #レイヤの生成
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftMaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    #入力データx 教師データt
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads
    
    def gradient(self, x, t):
        #forward
        self.loss(x, t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #設定
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads
"""
この実装ではニューラルネットワークのレイヤをOrderedDictに保持する点が重要
OrdereDictは順番付きのディクショナリである。ディクショナリに追加した要素の順番を覚えることができる。
よって、ニューラルネットワークの順伝播では追加した順にレイヤのforward()メソッドを呼び出すだけで処理が完了する
"""
##まとめ
"""
これまで勾配を求める方法を2つ用いた
一つは数値微分によって求める方法
もう一つは解析的に数式を解いて求める方法。
後者は誤差逆伝播法を用いることで大量のパラメータが存在しても効率的に計算できる。
よってこれからは数値微分を用いるのではなく誤差逆伝播法を用いて勾配を求めることにする。
とはいえ、数値微分は必要ないかと思われがちであるが、実は誤差逆伝播法を用いて実装し結果を出したとき、それが正しいものなのか、しっかり実装ができたのか
それを確かめる時に必要なものとなるので、しっかりと知る必要がある。
"""
