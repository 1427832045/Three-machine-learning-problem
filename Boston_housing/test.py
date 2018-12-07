from keras.datasets import boston_housing
from keras.models import load_model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data('./boston_housing.npz')
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
model = load_model('boson_housing_model.h5')
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)  # 查看误差
print('预测结果与实际平均相差', test_mae_score)
data = model.predict(test_data)
for j in range(5):
    print('预测值', data[j][0])
    print('实际值', test_targets[j])
