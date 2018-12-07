from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
from pyecharts import Scatter
from pyecharts import Line
from pyecharts import Overlap
from time import clock

clock()
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000, path='./reuters.npz')


def vect_seq(seqs, dimension=10000):
    results = np.zeros((len(seqs), dimension))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results


y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)  # 测试标签
layer1 = layers.Dense(64, activation='relu', input_shape=(10000,))
layer2 = layers.Dense(64, activation='relu')  # 自动推导输入形状
layer3 = layers.Dense(46, activation='softmax')
model = models.Sequential()
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
x_val = vect_seq(train_data[:1000])
partial_x_train = vect_seq(train_data[1000:])
y_val = y_train[:1000]
partial_y_train = y_train[1000:]
history = model.fit(x=partial_x_train, y=partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))  # 训练模型
print('训练用时', clock(), 's')
model.save('reuters_model.h5')  # 保存模型
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = list(range(1, len(acc) + 1))
"""pyecharts画图"""
overlap = Overlap('Traning acc and Validation acc')  # 将不同类型图表画在一张图上
scatter = Scatter()
scatter.add('Traning acc', epochs, acc)
overlap.add(scatter)
line = Line()
line.add('Validation acc', epochs, val_acc, is_smooth=True)
overlap.add(line)
overlap.render()  # 训练集和验证集的准确率变化
