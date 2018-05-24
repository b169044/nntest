#-*- coding: UTF-8 -*-

from sklearn import datasets
import math
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid_T(yhat):
    return (1-yhat)*yhat

def tanh_T(yhat):
    return 1-yhat**2

def get_wb(last_node_number,this_node_number,n = 0.01):
    w = np.random.random(last_node_number*this_node_number)*n
    w = np.reshape(w,(last_node_number,this_node_number))
    b = np.random.random(this_node_number)*n
    b = np.reshape(b,(1,this_node_number))
    return w,b

def dj2dz(y,yhat):
    return (1-y)/(1-yhat)-y/yhat

iris = datasets.load_iris()['data'][0:100]
y = datasets.load_iris()['target'][0:100]
x = iris.T
y = np.reshape(y,(1,-1))
alpat = 0.05
m = x.shape[1]
iterations_number=25000

# 隐藏层每个层中节点个数
lay_node_number = [3]

# 隐藏层和输出层的激活函数和激活函数的反函数
func = [tanh,sigmoid]
funct = [tanh_T,sigmoid_T]

# # 隐藏层每个层中节点个数
# lay_node_number = [10,3]

# # 隐藏层和输出层的激活函数和激活函数的反函数
# func = [tanh,tanh,sigmoid]
# funct = [tanh_T,tanh,sigmoid_T]

# 加入输入层和输出层的节点个数
lay_node_number.insert(0,x.shape[0])
lay_node_number.append(1)
lay_number = len(lay_node_number)

# 初始化w和b
w = []
b = []
np.random.seed(1)
for i in range(lay_number-1):
    tmp_w,tmp_b = get_wb(lay_node_number[i],lay_node_number[i+1])
    w.append(tmp_w)
    b.append(tmp_b)

for j in range(iterations_number):
    # 正向计算
    x_cache = [x]
    for i in range(lay_number-1):
        z = np.dot(w[i].T,x_cache[i])+b[i].T
        a = func[i](z)
        x_cache.append(a)
    y_hat=x_cache[len(x_cache)-1]

    # 计算误差
    J = - np.sum(y * np.log(y_hat)+(1-y)*np.log(1-y_hat))/m
    if(j%1000 == 0):
        print("当前迭代次数:"+str(j)+"\t\t误差："+str(J))

    # 反向传播
    da = dj2dz(y,y_hat)
    for i in range(len(w)-1,-1,-1):
        dz = da * funct[i](x_cache[i+1])
        dx = np.dot(w[i],dz)/m
        dw = np.dot(x_cache[i],dz.T)/m
        db = np.sum(dz.T,axis=0,keepdims=True)/m
        w[i] = w[i] - alpat*dw
        b[i] = b[i] - alpat*db
        da = dx

# 预测结果
a = x
for i in range(lay_number-1):
    z = np.dot(w[i].T,a)+b[i].T
    a = func[i](z)

y_predict  = (a > 0.5 )+ 0
print("预测结果:"+str(y_predict)+"\n实际结果:"+str(y)+"\n预测准确率:"+str((1-np.sum(np.abs(y_predict-y))/m)*100)+"%")