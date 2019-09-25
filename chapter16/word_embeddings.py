from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=5000)

# 限定数据集的长度
x_train = sequence.pad_sequences(x_train, maxlen=500)
x_validation = sequence.pad_sequences(x_validation, maxlen=500)

# 构建嵌入层：输入句子的长度：500；词典大小：5000（频率最高的5000个）；词向量维度：500
Embedding(5000, 32, input_length=500)
