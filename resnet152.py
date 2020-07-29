#coding:utf8
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet152
import torch.nn as nn
import datetime
import pandas as pd
from load_data import DogCat
#from torchsummary import summary

# 1=dog 0=cat
path = r'C:\Users\caeit\Desktop\binru7_1\train'
train_data = DogCat(path,train=True)
train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True)
model = resnet152(pretrained=True)
model.fc = nn.Linear(2048,2)
model = model.cuda()
#summary(model,(3,224,224))
cost = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
start = datetime.datetime.now()
for i in range(20):
        correct = 0.0
        total = 25000
        running_loss = 0.0
        accuracy = 0.0
        print('-----epoch', i+1, '-----')
        for num, image in enumerate(train_dataloader):
            x_train, y_train = image
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            optimizer.zero_grad()
            output = model(x_train)
            loss = cost(output, y_train)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y_train).sum().item()
            if num%100 == 0:
                print(num*16, '/ 25000', 'loss:', running_loss, ',accuracy:{}%'.format((100*correct/25000)),'correct:%s'%correct)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('-----------Epoch:', i+1, ', one_epoch_loss:', running_loss, '-----------',',accuracy:%f'%(100*correct/25000))
print(datetime.datetime.now()-start)
torch.save(model, 'fine_tuning_20_epoch_dog_cat_resnet152.pkl')

#测试
model = torch.load('fine_tuning_20_epoch_dog_cat_resnet152.pkl')
model.eval()
model = model.cuda()
path = r'C:\Users\caeit\Desktop\binru7_1\test1'
test_data = DogCat(path,train=False,test=True)
test_dataloader = DataLoader(test_data,batch_size=16,shuffle=False,num_workers=0)
result = []
start = datetime.datetime.now()
for num, image in enumerate(test_dataloader):
    x_train, y_train = image
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    output = model(x_train)
    _, predicted = torch.max(output, 1)
    result.append(predicted)
print(datetime.datetime.now()-start)
label = []
for i in result:
    i = i.cpu()
    for j in i.data.numpy():
        label.append(j)
for j,i in enumerate(label):
    if i == 0:
        label[j] = 0.05
    else:
        label[j] = 0.95
id = [i for i in range(1,12501)]
dataframe = pd.DataFrame({'id':id,'label':label})
dataframe.to_csv("Submission.csv",index=False, sep=',')