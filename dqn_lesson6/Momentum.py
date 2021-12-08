"""
Momentumモデル(モーメンタム)
v ← av - μ*ΘL/ΘW
W ← W + v
"""

"""
SGDと同じく、Wは更新する重みパラメータであり、ΘL/ΘWはWに関する損失関数の勾配、μは学習係数を表す。モーメンタムではここでvが登場する
物体が勾配方向に力を受け、その力によって物体の速度を加算されるという物理法則に基づいたものである。
"""

import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}#インスタンス変数ｖは速度を保持する。最初はデータを保持しないがこのメソッドが呼び込まれるたびにパラメータと同じ構造のデータをディクショナリ変数として保持していく。
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
