# RBM-movie
## 基于受限玻尔兹曼机（RBM）的电影推荐系统实例
#### 前言
本项目代码是基于Deep Learning with TensorFlow IBM Cognitive Class ML0120EN的文章代码实现，
使用[http://grouplens.org/datasets/movielens/1m](http://grouplens.org/datasets/movielens/1m)的电影数据集进行训练，并对目标用户进行推荐

#### 目录
* 数据预处理（dat转csv）
* 数据预处理（csv转dict）
* RBM模型训练
* RBM模型预测（给目标用户推荐）

#### 快速开始
请自行下载数据集[http://grouplens.org/datasets/movielens/1m](http://grouplens.org/datasets/movielens/1m)，解压到data/目录中

* 数据预处理（dat转csv）
```
    python datprocess.py <dat path> <csv path>
    
    如：python datprocess.py ./data ./data
```   
    
* 数据预处理（csv转dict）
```
    python csvprocess.py <csv path> <dict path>
    
    如：python csvprocess.py ./data ./data
```    

* RBM模型训练
```
    python train.py <dict path> <model path>
    
    如：python train.py ./data ./data
```    

* RBM模型预测（给目标用户推荐）
```
    python train.py <dict path> <model path> <log path> <user id>
    
    如：python train.py ./data ./data ./data 1
```
#### 其他
我的github：https://github.com/JonLagrange
