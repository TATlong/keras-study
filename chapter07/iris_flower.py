from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import datasets
import numpy as np

"""
    多分类：鸢尾花分类  
        
"""

# 导入数据：这里的数据是sklearn中的自带数据集
dataset = datasets.load_iris()
X = dataset.data
Y = dataset.target


# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数：设置2个参数optimizer和init
def create_model(optimizer='adam', init='glorot_uniform'):
    # 序贯模型
    model = Sequential()
    # 输入层:4个神经元
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    # 输出层：3个神经元，采用softmax作为激活函数。
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 编译模型：采用adam优化器，用此优化器替换随机梯度下降的优化算法
    #         使用keras中称为分类交叉熵的对数损失函数
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 使用KerasClassifier封装分类问题的模型为scikit-learn中的基本模型，verbose=0：关闭模型中的log输出
model = KerasClassifier(build_fn=create_model,
                        epochs=200,
                        batch_size=5,
                        verbose=0)

# 10折交叉验证：在分割数据之前，使用shuffle对数据进行随机乱序排列
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# 交叉验证运行
results = cross_val_score(model, X, Y, cv=kfold)

# 输出均值和标准差
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))