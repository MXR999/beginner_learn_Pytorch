import os
from abc import ABC

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "True"  # 在cpu上运行
# 以上导入我也不知道为啥，但是不导入我的程序就报错，想了解可自行实验，查找原因

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise

# 用pytorch建模，pytorch默认都是对一个批次的数据进行训练
# 一个批次的数据是多维的
# 所以，这里改变一下数据的形状 (变成2维)
x_data = x_data.reshape(-1, 1)  # (-1, 1) 就是不管多少行，先固定 1列
y_data = y_data.reshape(-1, 1)
# print(x_data.shape)  # 可以查看一下数据形状  (100, 1)
# print(y_data.shape)  # 可以查看一下数据形状  (100, 1)

# x_data, y_data 都是 numpy 生成的数据
# 需要转换成pytorch可用的tensor类型
x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)

# 再将数据变成 Variable 变量
inputs = Variable(x_data)  # model的输入
target = Variable(y_data)  # model的label


# 构建 neural network model
# 一般把网络中有可学习参数的层放在初始化中
class LinearRegression(nn.Module):  # 定义一个类，类的名字随便取，最好是你的 model name
    # 初始化的目的就是要定义 网络的结构
    def __init__(self):
        # 初始化父类 nn.Module （固定操作，可以记下来）
        super(LinearRegression, self).__init__()
        # 定义全连接层 （此网络很简单，只有1个输入，1个输出）
        self.fc = nn.Linear(1, 1)  # mouse悬停在Linear，按住ctrl+单击mouse左键，可以查看Linear的参数信息

    # 定义前向传播，也就是整个网络的计算顺序
    def forward(self, x):  # x就是model的输入
        # out 就是经过全连接层fc的输出，也就是这个简单model的输出
        out = self.fc(x)
        # 返回这个输出
        return out


# 不用定义backward()，pytorch可以自动求解参数

# 现在要开始使用我们创建的模型啦
# 实例化model
my_model = LinearRegression()

# 定义损失函数 （均方差loss）
mse_loss = nn.MSELoss()

# 定义优化器 （随机梯度下降法）
# （传入my_model的参数，lr就是learning rate）
optimizer = optim.SGD(my_model.parameters(), lr=0.1)

# - - - - - - - - - star
# 这一部分跟整个模型的定义无关，只是用来查看my_model的参数，你可以自行注释掉
for name, parameters in my_model.named_parameters():
    # for循环，得到model的 name 和 参数
    print(f'name:{name}, para:{parameters}')
# - - - - - - - - - end

# model的训练  循环训练1000次
for i in range(1001):
    out = my_model(inputs)
    # 计算loss
    loss = mse_loss(out, target)
    # 在backward()之前，要先梯度清0
    optimizer.zero_grad()
    # 计算gradient
    loss.backward()
    # 更新参数，修改权值
    optimizer.step()
    # 观察参数的更新状态 （每200次打印一次loss的值）
    if i % 200 == 0:
        print(f"第{i}次训练，loss={loss.item()}")  # loss.item() 把loss的类型转换成python的类型

# 观察model的效果
plt.scatter(x_data, y_data)
y_pred = my_model(inputs)
# y_pred.data.numpy() 把tensor数据类型转换成numpy数据类型
# ‘r-’ red的线， lw=3 线宽为3
plt.plot(x_data, y_pred.data.numpy(), 'r-', lw=2)
plt.show()


