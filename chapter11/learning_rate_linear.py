from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

"""
    学习率衰减:
    （1）线性衰减（根据epoch逐步降低学习率）
    （2）指数衰减（在特定epoch使用分数快速降低学习率）
"""

# 导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 模型优化：线性衰减率公式：learning Rate = (learning Rate) * (1/(1+decay*epoch))
    learningRate = 0.1

    # 动量值为0.9:使用较大的动量值有助于优化算法在学习率缩小到很小的值时，继续向正确的方向更新权重。
    momentum = 0.9
    decay_rate = 0.005
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


epochs = 200
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=5, verbose=1)
model.fit(x, Y)