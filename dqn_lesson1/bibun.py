def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt
import griddata
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import gradient
from gradient_2d import numerical_gradient


#x = np.arange(0.0,20.0,0.1)
#y = function_1(x)
#plt.xlabel("x")
#plt.ylabel("f(x)")
#plt.plot(x,y)
#plt.show()

def function_2(x):#偏微分
    if x.ndim == 1: #関数が１次元の場合
        return np.sum(x**2)
    else:
        return np.sum(x**2,axis=1)

#fig = plt.figure()
#ax = Axes3D(fig)

#X = np.arange(-3.0,3.0,0.1)
#Y = np.arange(-3.0,3.0,0.1)
#X,Y = np.meshgrid(X,Y)
#Z = X**2+Y**2

#ax.set_xlabel("x0")
#ax.set_ylabel("x1")
#ax.set_zlabel("f(x)")

#ax.set_xlim(-3.0,3.0)
#ax.set_ylim(-3.0,3.0)
#ax.set_zlim(0.0,18.0)

#ax.view_init(25,-120)
#ax.plot_wireframe(X,Y,Z)
#plt.show()

#勾配
def numerical_gradient_1ndim(f,x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x)#xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h)の計算
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2)/(2*h)
        
        x[idx] = tmp_val
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_1ndim(f, X)#1次元の場合
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1ndim(f, x)
        
        return grad
    
#x0 = np.arange(-2.0, 2.5, 0.25)
#x1 = np.arange(-2.0, 2.5, 0.25)
#X, Y = np.meshgrid(x0, x1)

#X=X.flatten()
#Y=Y.flatten()

#grad = numerical_gradient(function_2,np.array([X,Y]).T).T

#plt.figure()
#plt.quiver(X,Y,-grad[0],-grad[1],angles="xy",color="#666666")
#plt.xlim([-2,2])
#plt.ylim([-2,2])
#plt.xlabel("x0")
#plt.grid()
#plt.draw()
#plt.show()

#print(numerical_gradient(function_2,np.array([3.0,4.0])))

#勾配降下法
def gredientdescent(f,init_x,lr=0.01,step_num=100):
    x = init_x #引数fは最適化したい関数であり、init_xは初期値、lrは学習率
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f,x)#関数の勾配を求める
        x -= lr * grad#求めた勾配より学習率を掛けていく(ゆくゆくは最小値となる)

    return x,np.array(x_history)

def function_3(x):
    return x[0]**2+x[1]**2

#init_x = np.array([-3.0,4.0])
#print(gredientdescent(function_3, init_x=init_x, lr=0.1, step_num=100))
#lr = 0.1
#step_num = 20
#x,x_history = gredientdescent(function_3,init_x=init_x,lr=lr,step_num=step_num)

#plt.plot([-5,5],[0,0],"--b")
#plt.plot([0,0],[-5,5],"--b")
#plt.plot(x_history[:,0],x_history[:,1],"o")

#plt.xlim(-3.5,3.5)
#plt.ylim(-4.5,4.5)
#plt.xlabel("X0")
#plt.ylabel("X1")
#plt.show()