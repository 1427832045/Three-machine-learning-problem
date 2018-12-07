from keras.datasets import boston_housing
from keras import models
from keras import layers
from time import clock

clock()
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data('./boston_housing.npz')
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std


def build_model():
    """
    构建神经网络
    :return: 模型
    """
    layer1 = layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],))
    layer2 = layers.Dense(64, activation='relu')  # 自动推导输入形状
    layer3 = layers.Dense(1)
    model = models.Sequential()
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


"""k折验证"""
# k = 4
# num_val_samples = len(train_data) // 4
# num_epochs = 50  # 训练轮数
# all_mae_history = []
# for i in range(k):
#     print('p............', i)
#     val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]  # 数据切片，验证数据，第k个分区的数据
#     val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]
#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples],
#          train_data[(i + 1) * num_val_samples:]],
#         axis=0
#     )
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i + 1) * num_val_samples:]],
#         axis=0
#     )
#     model = build_model()
#     history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
#                         validation_data=(val_data, val_targets), batch_size=1, verbose=0)  # 静默模式 ，verbose=0
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_history.append(mae_history)
# average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
# # line = Line()
# # line.add(list(range(1, len(average_mae_history) + 1)),average_mae_history,is_symbol_show=False)
# # line.render()
# # print(clock())
# plt.plot(range(1, len(average_mae_history) + 1),average_mae_history,)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
model.save('boson_housing_model.h5')
print('用时', clock(), 's')
