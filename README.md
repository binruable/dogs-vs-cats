# An example of image recognition(kaggle competition:dogs vs cats)

## Pytorch implementation

## Preprocess

* 每张图片压缩到[0,1]（toTensor）
* 减去mean=[0.485, 0.456, 0.406]，再除以std = [0.229, 0.224, 0.225]（归一化）

## Data

* download: https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip

## Result

|                    | running time(GPU:2080Ti) | pretrain |  accruay（train: 25000，test: 12500）   |
| :----------------: | :----------------------: | :------: | :-------------------------------------: |
| vgg16（epoch：20） |         1:17:57          |   True   | train:  98.98% ；test(logloss): 0.12360 |
| vgg19（epoch：20） |                          |          |                                         |
|                    |                          |          |                                         |
|                    |                          |          |                                         |

* 训练评价指标使用LogLoss

$$\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right]$$
