from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


""" 
    dropout()使用：
    （1）通常设置为20%-50%的dropout率，太低了产生的作用很小，太高的话可能导致网络训练的不充分。
    （2）在较大的网路效果可能更好。
    （3）使用较高的动量值提高学习率，但高的学习率会导致大的权重。
    （4）限制网络权重的大小，使用正则化，在keras中使用dense的kernel_constraint=maxnorm(3)来限制网络权重。
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
    # 定义Dropout
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model,
                        epochs=200,
                        batch_size=5,
                        verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))