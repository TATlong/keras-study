import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pandas import read_csv

"""
    二分类实例：银行营销分类
    该例子使用UCI机器学习仓库中的Banking Marking的数据集
    （1）数据格式化：神经网络模型对规模和分布一致的输入值有很好的学习效果。
        数据标准化是对数据进行缩放，使每个属性的平均值为0，标准偏差为1，且使数据保持高斯分布。
        又叫做数据的正态化
"""

# 导入数据并将分类转化为数字
dataset = read_csv('bank.csv', delimiter=';')
dataset['job'] = dataset['job'].replace(to_replace=['admin.', 'unknown', 'unemployed', 'management',
                                                    'housemaid', 'entrepreneur', 'student', 'blue-collar',
                                                    'self-employed', 'retired', 'technician', 'services'],
                                        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['marital'] = dataset['marital'].replace(to_replace=['married', 'single', 'divorced'], value=[0, 1, 2])
dataset['education'] = dataset['education'].replace(to_replace=['unknown', 'secondary', 'primary', 'tertiary'],
                                                    value=[0, 2, 1, 3])
dataset['default'] = dataset['default'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['housing'] = dataset['housing'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['loan'] = dataset['loan'].replace(to_replace=['no', 'yes'], value=[0, 1])
dataset['contact'] = dataset['contact'].replace(to_replace=['cellular', 'unknown', 'telephone'], value=[0, 1, 2])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace=['unknown', 'other', 'success', 'failure'],
                                                  value=[0, 1, 2, 3])
dataset['month'] = dataset['month'].replace(to_replace=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                            value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
dataset['y'] = dataset['y'].replace(to_replace=['no', 'yes'], value=[0, 1])


# 分离输入输出
array = dataset.values
X = array[:, 0:16]
Y = array[:, 16]

# 设置随机种子
seed = 7
np.random.seed(seed)


# 构建基础模型函数：默认值将作为生成基准模型的参数
def create_model(units_list=[16], optimizer='adam', init='normal'):
    # 序列模型
    model = Sequential()
    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=16, kernel_initializer=init))
    # 构建更多隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# 包装类封装模型
model = KerasClassifier(build_fn=create_model,
                        epochs=200,
                        batch_size=5,
                        verbose=0)

# k折交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))

"""
数据预处理后，训练模型
"""

# 数据标准化处理
new_x = StandardScaler().fit_transform(X)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, new_x, Y, cv=kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))


"""
网格搜索最优模型参数
"""

# 调参选择最优模型
param_grid = {}
param_grid['units_list'] = [[16], [30], [16, 8], [30, 8]]
# 调参
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(new_x, Y)

# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, std, param))
