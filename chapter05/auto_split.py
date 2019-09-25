from keras.models import Sequential
from keras.layers import Dense
import numpy as np

"""
    学习：
    （1）不同于手动划分数据集，这里是直接在训练模型时model.fit()中指定验证集的大小，这里并没有指定测试集

"""


# 设定随机数种子：使用固定随机数种子初始化随机生成器，这样就可以重复的运行相同的代码，并得到相同的结果。
np.random.seed(7)

# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# 分割输入x和输出Y
X = dataset[:, 0: 8]
Y = dataset[:, 8]

# 创建模型:序贯模型
model = Sequential()

# input_dim：创建第一层，8代表有8个输入变量，也就是输入数据的维度，第一层需要定义输入的数据，后面就不需要；
# Dense：定义全连接层
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型：必须指定损失函数，在这里是对数损失函数，对于二进制分类问题的对数损失函数被定义为二进制交叉熵。
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型并自动评估模型
model.fit(x=X, y=Y,
          epochs=150,
          batch_size=10,
          validation_split=0.2)

