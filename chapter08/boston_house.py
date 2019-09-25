from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import numpy as np


"""
    回顾实例：波士顿房价预测
    （1）数据集中数据的各个属性（维度）的尺度不同，因此在使用模型之前需要对数据进行预处理。
    
"""


# 导入数据
dataset = datasets.load_boston()
X = dataset.data
Y = dataset.target

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(units_list=[13], optimizer='adam', init='normal'):
    # 序贯模型
    model = Sequential()
    # 构建第一个隐藏层和输入层
    units = units_list[0]
    model.add(Dense(units=units, activation='relu', input_dim=13, kernel_initializer=init))
    # 构建更多隐藏层:这里是通用的设置
    for units in units_list[1:]:
        model.add(Dense(units=units, activation='relu', kernel_initializer=init))
    # 最后一层
    model.add(Dense(units=1, kernel_initializer=init))

    # 编译模型：这里loss采用均方误差（MSE）
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


model = KerasRegressor(build_fn=create_model,
                       epochs=200,
                       batch_size=5,
                       verbose=0)

# 设置算法评估基准
kfold = KFold(n_splits=10,
              shuffle=True,
              random_state=seed
              )

# 开始处理
# results = cross_val_score(model, X, Y, cv=kfold)
# print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))


"""数据正态化，改进算法
   这里使用sklearn.pipeline框架
"""

steps = []

# 数据的标准化处理
steps.append(('standardize', StandardScaler()))
steps.append(('mlp', model))
pipeline = Pipeline(steps)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# 开始处理
results = cross_val_score(pipeline, X, Y, cv=kfold)
print('Standardize: %.2f (%.2f) MSE' % (results.mean(), results.std()))


"""
网格搜索最优参数
"""

# 调参选择最优模型
param_grid = {}
param_grid['units_list'] = [[20], [13, 6]]
param_grid['optimizer'] = ['rmsprop', 'adam']
param_grid['init'] = ['glorot_uniform', 'normal']
param_grid['epochs'] = [100, 200]
param_grid['batch_size'] = [5, 20]


# 调参
scaler = StandardScaler()

# 数据标准化处理
scaler_X = scaler.fit_transform(X)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
results = grid.fit(scaler_X, Y)


# 输出结果
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, std, param))


"""
Baseline: -22.58 (11.30) MSE
Standardize: -13.06 (6.93) MSE
"""