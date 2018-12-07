from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
from time import clock

clock()
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=50000,
                                                                      path='./imdb.npz')  # 获得训练评论和测试评论各25000条，仅其中保留50000个高频词汇


def vect_seq(seqs, dimension=50000):
    results = np.zeros((len(seqs), dimension))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results


x_test = vect_seq(test_data)  # 测试集
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')  # 测试标签
layer1 = layers.Dense(16, activation='relu', input_shape=(50000,))
layer2 = layers.Dense(16, activation='relu')  # 自动推导输入形状
layer3 = layers.Dense(1, activation='sigmoid')
model = models.Sequential()
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
x_val = vect_seq(train_data[:10000])
partial_x_train = vect_seq(train_data[10000:])
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(x=partial_x_train, y=partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))  # 训练模型
"""
x：输入数据。如果模型只有一个输入，那么x的类型是numpy 
array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
y：标签，numpy array
batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
"""
print('用时', clock(), 's')
