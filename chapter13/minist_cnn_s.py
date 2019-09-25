from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend

backend.set_image_data_format('channels_first')


# 设定随机种子
seed = 7
np.random.seed(seed)

# 从Keras导入Mnist数据集
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')

# 格式化数据到0-1之前
X_train = X_train / 255
X_validation = X_validation / 255

# one-hot编码
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)


# 创建模型
def create_model():
    model = Sequential()

    # 第一个隐藏层：5x5的感受野，32个特征图，输入1通道的28x28，relu激活函数
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # 最大池化层，大小2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # dropout的正则化层，排除掉20%的神经元
    model.add(Dropout(0.2))

    # 将多维数据转化为一维数据的flatten层
    model.add(Flatten())

    # 全连接层
    model.add(Dense(units=128, activation='relu'))

    # 最后一层，10个神经元，10个分类
    model.add(Dense(units=10, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()

# verbose=2，仅输出每个epoch的最终结果，忽略每个epoch的详细内容
model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=2)

score = model.evaluate(X_validation, y_validation, verbose=0)
print('CNN_Small: %.2f%%' % (score[1] * 100))