from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


"""
    在训练深度学习模型时：
        （1）可以利用检查点来捕获模型的权重
        （2）可以基于当前的权重进行预测
        （3）可以使用检查点保存的权重继续训练模型
    
    keras中的callbacks：
        （1）提供检查点的功能
        （2）指定要监视的指标,例如:训练或评估数据集的丢失或准确率；
                               也可以指定是否寻求最大或最小化的改进；
                               用于存储权重的文件名可以包括诸如epochs编号或评估矩阵的变量
    
    ModelCheckpoint回调类：
        （1）可以定义模型的权重值检查点的位置、文件的名称
        （2）以及在什么情况下创建模型的检查点
    
"""


# 导入数据
dataset = datasets.load_iris()
X = dataset.data
Y = dataset.target

# Convert labels to categorical one-hot encoding
Y_labels = to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 构建模型
model = create_model()


# 设置检查点
filepath = "./checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"

# 只有在评估数据集（monitor="val_acc" 和 model = "max"）上的分类准确度有所提高时才会设置检查点
# 来保存网络权重。
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             mode='max',
                             verbose=1,
                             save_best_only=True
                             )

# 加入列表
callback_list = [checkpoint]

# 开始训练
model.fit(X, Y_labels,
          validation_split=0.2,    # 使用20%的数据自动评估模型性能
          epochs=200,
          batch_size=5,
          verbose=0,
          callbacks=callback_list)

