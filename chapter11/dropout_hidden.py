from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


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
    model.add(Dense(units=4,
                    activation='relu',
                    input_dim=4,
                    kernel_initializer=init,
                    kernel_constraint=maxnorm(3))
              )

    # 输入层添加dropout层，作为一种添加噪声的方法。设置为20%，每个更新周期将会被随机排除。
    model.add(Dropout(rate=0.2))

    model.add(Dense(units=6,
                    activation='relu',
                    kernel_initializer=init,
                    kernel_constraint=maxnorm(3))
              )

    # 隐层的dropout层
    model.add(Dropout(rate=0.2))

    model.add(Dense(units=3,
                    activation='softmax',
                    kernel_initializer=init)
              )

    # 定义优化器：
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


# 包装器封装模型
model = KerasClassifier(build_fn=create_model,
                        epochs=200,
                        batch_size=5,
                        verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))