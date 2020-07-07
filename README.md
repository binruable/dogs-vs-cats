# An example of image recognition(kaggle competition:dogs vs cats)

## Pytorch implementation

## Preprocess

* 每张图片压缩到[0,1]（toTensor）
* 减去mean=[0.485, 0.456, 0.406]，再除以std = [0.229, 0.224, 0.225]（归一化）

## Data

* download: https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip
![](./doc/222.png)

## Result

|              | epochs | running time(GPU:2080Ti) | pretrain |  accruay（train: 25000，test: 12500）   |
| :----------: | :----: | :----------------------: | :------: | :-------------------------------------: |
|    VGG16     |   20   |         1:17:57          |   True   | train:  98.98% ；test(logloss): 0.12360 |
|    VGG19     |   20   |                          |   True   |                                         |
|  ResNet152   |   20   |                          |          |                                         |
| Inception-v4 |   20   |                          |          |                                         |

* 训练评价指标使用LogLoss
![](./doc/111.png)