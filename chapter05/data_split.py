from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

"""
    学习：
    （1）numpy的loadtxt()读取.csv文件
    （2）numpy设定随机数种子
    （3）sklearn数据集的划分（x_train, x_validation），这里没有指定测试集
    （4）Sequential()序贯构建深度模型
    （4）简单的模型参数设置
"""


seed = 7

# 设定随机数种子
np.random.seed(seed)

# 导入数据:使用的是numpy的loadtxt，感觉很好用啊
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# 分割输入x和输出Y：这时样本和标签都是一一对应的。
X = dataset[:, 0:8]
Y = dataset[:, 8]

# sklearn的model_selection分割数据集：返回数据的格式是固定的，不能调换顺序
x_train, x_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=seed)

# 构建模型：序贯模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型：损失函数和优化器的设置
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型：训练数据和训练标签，验证数据和标签
model.fit(x_train, Y_train,
          epochs=150,
          batch_size=10,
          validation_data=(x_validation, Y_validation)
          )


"""
说明:
Train on 614 samples, validate on 154 samples
从参数设置和训练结果可以看出，总共是150次epochs，在每个epochs中，每批的训练数据是10条样本，
训练数据是614条，验证数据是154条。

"""