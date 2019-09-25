from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json

"""
    学习：
        模型保存：（1）模型结构描述文件 （2）模型权重文件
"""

# 导入数据
dataset = datasets.load_iris()
X = dataset.data
Y = dataset.target
print(len(Y), Y)

# 将标签转换为one-hot编码
Y_labels = to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # 序列模型构建
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 构建模型
model = create_model()
model.fit(X,
          Y_labels,
          epochs=200,
          batch_size=5,
          verbose=0)

scores = model.evaluate(X, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

# 模型保存成Json文件
model_json = model.to_json()
with open('./model_json/model.json', 'w') as file:
    file.write(model_json)
# 保存模型的权重值
model.save_weights('./model_json/model.json.h5')


# 从Json加载模型
with open('./model_json/model.json', 'r') as file:
    model_json = file.read()

# 加载模型
new_model = model_from_json(model_json)
new_model.load_weights('./model_json/model.json.h5')

# 编译模型
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 评估从Json加载的模型
scores = new_model.evaluate(X, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
