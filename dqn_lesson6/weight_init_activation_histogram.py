"""
隠れ層のアクティベーションの分布を観察することで多くの知見がみられる。
ここでは重みの初期値によって隠れ層のアクティベーションがどのような変化するのか実験する。
そもそも、これは重みの初期設定がニューラルネットワークの学習において最も重要だからだ
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

x = np.random.randn(1000, 100)#1000個のデータ
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[1-1]

    #W = np.random.randn(node_num, node_num) * 0.01

    #Xavierharの初期値は一般的なディープラーニングのフレームワークで標準的に用いられている。具体的には前層のノードの個数をnとした場合、1/√nの標準偏差を持つ分布を使うということである。
    W = np.random.randn(node_num, node_num) / np.sqrt(node_num)#Xavierの初期値を適用

    z = np.dot(x, W)
    a = sigmoid(z) #シグモイド関数
    activations[i]=a

"""
ここでは5つの層がある。それぞれの層は100個のニューロンを持つものとし、入力データとして1000個のデータをガウス分布でランダムに生成し、
それを5層ニューラルネットワークに流すというものである。

ではactivationsに格納された各層のデータをヒストグラムとして描画してみる
"""

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

    