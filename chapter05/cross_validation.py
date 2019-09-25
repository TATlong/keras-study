from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold


"""
    学习：
    （1）k折交叉验证
    （2）标准差 
        
"""
seed = 7

# 设定随机数种子
np.random.seed(seed)

# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# 分割输入x和输出Y
X = dataset[:, 0:8]
Y = dataset[:, 8]

# K折交叉验证：StratifiedKFold是KFold的变体
# StratifiedKFold通过算法平衡每个子集中每个类的实例数，这里分割成10个子集
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
cvscores = []

print(len(X))
# 这里把数据集分割成10个子集，得到10个模型
for train, validation in kfold.split(X, Y):
    print(len(X[train]), len(X[validation]))

    # 创建模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型：通过设置verbose=0关闭模型的fit()和evaluate()函数的详细输出。
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)

    # 评估模型
    scores = model.evaluate(X[validation], Y[validation], verbose=0)

    # 输出评估结果
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)


# 输出均值(np.mean)和标准差(np.std)
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))
