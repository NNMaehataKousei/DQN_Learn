#勾配確認
"""
誤差逆伝播法を用いれば効率よく計算できるが複雑な構造をしている。そのため入力ミスなどによる正しく勾配されていない可能性を確認するために、
実装が簡単で見やすいが時間がかかる数値微分を使って出した出力を誤差伝播法で用いた出力と比較し、近似しているかどうかを確認する
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

#データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + "." + str(diff))
