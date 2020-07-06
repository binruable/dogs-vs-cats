# An example if image recognition

### Preprocess

* 每张图片压缩到[0,1]减去mean=[0.485, 0.456, 0.406]再除以std = [0.229, 0.224, 0.225]（归一化）

## Data

* download: https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip

## Result

|                    | fine tuning | accruay（train: 25000，test: 12500） |
| :----------------: | :---------: | ------------------------------------ |
| vgg16（epoch：20） |             |                                      |
| vgg19（epoch：20） |             |                                      |
|                    |             |                                      |



