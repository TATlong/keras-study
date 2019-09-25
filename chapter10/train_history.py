from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt

"""
    keras提供了对历史训练数据的默认回调方法：
        在深度学习的模型训练过程中，默认回调方法之一是history回调，它记录了每个epoch的训练指标。
        模型的训练过程中的信息可以从fit()函数中返回。
        
       （1）模型在epoch上的收敛速度（斜率）
       （2）模型是否已经收敛（该线是否平滑收敛）
       （3）模型是否过度学习训练数据（验证线的拐点）
"""

# 导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target

# Convert labels to categorical one-hot encoding
Y_labels = to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 构建模型
model = create_model()

# model.fit()保存在了history对象中
History = model.fit(x, Y_labels,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=5,
                    verbose=0)

# 评估模型
scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

# Hisotry列表
print(History.history.keys())

# accuracy的历史
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# loss的历史
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
