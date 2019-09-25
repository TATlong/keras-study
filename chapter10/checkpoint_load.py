from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

"""
神经网络的拓扑结构可以在训练模型前，序列化json或者yaml格式的文件，以确保可以恢复网络的拓扑结构，
在下面的例子中，模型的拓扑结构是已知的，直接从加载权重文件
"""

# 导入数据
dataset = datasets.load_iris()
X = dataset.data
Y = dataset.target

# Convert labels to categorical one-hot encoding
Y_labels = to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def load_model(optimizer='rmsprop', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 加载权重
    filepath = './checkpoint_overwrite/weights.best.h5'
    model.load_weights(filepath=filepath)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 加载模型
model = load_model()

# 评估模型
scores = model.evaluate(X, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

