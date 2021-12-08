"""
今回の目的
ニューラルネットワークの学習の目的は、損失関数の値をできるだけ小さくするパラメータを見つけることです。これは最適なパラメータを見つける問題であり
そのような問題を解くことを最適化という。この最適化が実に難しい。今回は確率的勾配降下法によるパラメータの最適化を実装する。そして
確率的勾配降下法の欠点をみる。それを理解したうえでよりよい手法へ着手とする

SGDのモデル
W　←　W  - μ*ΘL/ΘW
"""
class SGD:#確率的勾配降下法をクラスで定義
    def __init__(self, lr=0.01):
        self.lr = lr #lrは学習係数を表す

    def update(self, params, grads):#SGDではこのアップデートメソッドを繰り返し読み込まれることになる
        #paramsとgradsはニューラルネットワークのディクショナリ変数
        for key in params.keys():
            params[key] -= self.lr * grads[key]

#このSGDを利用した場合のニューラルネットワークは次のようなコードの構造で実装できる。
"""
network = TwoLayerNet(....)#レイヤを定義
optimizer = SGD()#確率的勾配降下法のSGDを定義

for i in range(10000):
    x_batch, t_batch = get_mini_batch(....)#ミニバッチ
    grads = network.gradient(x_batch, t_batch)#偏微分
    params = network.params
    optimizer.update(params, grads)

SGDにはパラメータと勾配の情報を渡すだけ
"""
"""
SGDの欠点ではお椀のような形でx軸に伸びた形状をしたモデルには向いていないということである。
"""



    