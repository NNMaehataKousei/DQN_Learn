"""
ニューラルネットワークでは学習係数の値が重要となる。大きければ発散するし、小さければ時間がかかる。
そこで最初は大きく値を設定し、だんだん値を小さくする手法がある。それがAdagradというものである。

数式モデルは
h ← h + ΘL/ΘW * ΘL/ΘW
W ← W - μ*1/√h*ΘL/ΘW
SGDと同じく、Wは更新する重みパラメータ、ΘL/ΘWはWに関する損失関数の勾配、μは学習係数
今回は新たにhを用意している。これはこれまでに経験した勾配の値を2乗和として保持するものである。
そしてパラメータが更新するにつれて1/√hを乗算することで学習スケールを調整するのだ。
これはパラメータの要素の中でよく動いた（大きく更新された）要素は、学習係数が小さくなることを意味する。
"""
import numpy as np
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)#self,h[key]が0だった場合、除算されないので1e-7と設定している。1e-7は今回限りでこの数値でないといけないわけではないことに注意
    
"""
Adamという手法もあるが、これはAdamGradとmomentumを合わせたようなものである。今回は省く
"""