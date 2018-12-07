# IMDB Movie reviews sentiment classification

## 介绍
### 目的
经典的机器学习二分类问题，使用来自IMDB的25,000条影评，运用keras训练神经网络将影评划分为正面/负面两种评价
### 文件
fit.py为训练模型的脚本，test.py为测试脚本，my_model.h5为训练好的模型（可以直接使用），imdb.npz为影评数据文件



## 使用模型




```python
model = load_model('my_model.h5')
```
