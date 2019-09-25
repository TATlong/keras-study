import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

seed = 7
batch_size = 1
epochs = 100
filename = 'international-airline-passengers.csv'
footer = 3
look_back = 1


def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


def build_model():
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(1, look_back)))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':

    # 设置随机种子
    np.random.seed(seed)

    # 导入数据
    data = read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
    dataset = data.values.astype('float32')

    # 标准化数据：使用MinMaxScaler预处理类对数据集进行归一化处理，因为LSTM对输入数据的尺度敏感。
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # 训练数据集和验证数据集的大小构造
    train_size = int(len(dataset) * 0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

    # 创建dataset，让数据产生相关性
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)

    # 将输入转化成为[sample， time steps, feature]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))

    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # 模型预测数据
    predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)

    # 反标准化数据 --- 目的是保证MSE的准确性
    predict_train = scaler.inverse_transform(predict_train)
    y_train = scaler.inverse_transform([y_train])
    predict_validation = scaler.inverse_transform(predict_validation)
    y_validation = scaler.inverse_transform([y_validation])

    # 评估模型
    train_score = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    validation_score = math.sqrt(mean_squared_error(y_validation[0], predict_validation[:, 0]))
    print('Validatin Score: %.2f RMSE' % validation_score)

    # 构建通过训练集进行预测的图表数据
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train

    # 构建通过评估数据集进行预测的图表数据
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[len(predict_train) + look_back * 2 + 1:len(dataset) - 1, :] = predict_validation

    # 图表显示
    dataset = scaler.inverse_transform(dataset)
    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_validation_plot, color='red')
    plt.show()
