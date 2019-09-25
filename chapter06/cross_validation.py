from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


seed = 7

# 设定随机数种子
np.random.seed(seed)


# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]


def create_model():
    # 构建模型
    model = Sequential()
    model.add(Dense(units=12, input_dim=8, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 返回模型
    return model


# 建模型 for scikit-learn
# build_fn：指定创建模型的函数名名称，参数将自动绑定并传递给KerasClassifier内部的fit()函数
model = KerasClassifier(build_fn=create_model,
                        epochs=150,
                        batch_size=10,
                        verbose=0)


# 10折交叉验证：把数据分割成10个数据子集
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# 该函数用来评估深度学习模型，并输出结果
results = cross_val_score(model, X, Y, cv=kfold)

# 模型的准确率平均值
print(results.mean())

