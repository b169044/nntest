#-*- coding: UTF-8 -*-

from  sklearn  import datasets
import numpy as np
# 获取计算数据
iris = datasets.load_iris()
x = iris['data'][:100].T
y = iris['target'][:100].T
m = x.shape[1]
alpat = 0.005
iterations_number = 20000
# 初始化w和b
np.random.seed(1)
w = (np.random.random(x.shape[0])*0.01).reshape(x.shape[0],1)
b = 0
for i in range(iterations_number):
    # 正向传递
    z = np.dot(w.T,x)+b
    a = 1/(1+np.exp(-z))
    # 计算误差
    J = - np.sum(y * np.log(a)+(1-y)*np.log(1-a))/m
    if(i%10000 == 0):
        print("当前迭代次数"+str(i)+"\t误差："+str(J))
    # 反向传递
    dw = np.dot(x,(a-y).T)/m
    db = np.sum(a-y)/m
    w = w - alpat*dw
    b = b - alpat*db

# 预测数据
z = np.dot(w.T,x)+b
a = 1/(1+np.exp(-z))
y_predict  = (a > 0.5 )+ 0

print("预测结果:"+str(y_predict)+"\n实际结果:"+str(y)+"\n预测准确率:"+str((1-np.sum(np.abs(y_predict-y))/m)*100)+"%")