# coding:utf8
import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils import data


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        np.random.seed(100)
        self.imgs = imgs
        if transforms is None:
            if self.test or not train:
                self.transforms = T.Compose([
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                imgs = np.random.permutation(imgs)
                self.transforms = T.Compose([
                    T.CenterCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('\\')[-1])
        else:
            label = 1 if 'dog' in img_path.split('\\')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
