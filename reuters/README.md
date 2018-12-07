# Reuters newswire topics classification

## 介绍
### 目的
经典的机器学习多分类问题，使用来自路透社的11,228条新闻，运用keras训练神经网络将影评划分为分为46个主题
### 文件
fit.py为训练模型的脚本，test.py为测试脚本，reuters_model.h5为训练好的模型（可以直接使用）



## 使用模型




```python
model = load_model('reuters_model.h5')
```
