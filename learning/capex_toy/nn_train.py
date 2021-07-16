from icecream import ic
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter()

path = os.getcwd()  # use your path
ic(path)

li_dem = []
li_gen = []
li_res = []
li_vom = []
for counter in range(0, 999):
    #df = pd.read_csv(path + '/data' + '/dem_data_%d.csv' % counter, usecols=['dem'])
    #li_dem.append(df['dem'].to_numpy())
    df = pd.read_csv(path + '/capex_toy/gen_data_%d.csv' % counter, usecols=['ccost', 'fomcost', 'cap'])
    li_gen.append(df.to_numpy().flatten())
    df = df = pd.read_csv(path + '/capex_toy/gen_data_%d.csv' % counter, usecols=['vomcost'], skiprows=[6,7])
    li_vom.append(df.to_numpy().flatten())
    df = pd.read_csv(path + '/capex_toy/result_cap_%d.csv' % counter, usecols=['cap'], skiprows=[6])
    # df = pd.read_csv(path+'/gen_result_%d.csv' % counter, usecols = ['g','gen'])
    # d = {}
    # for i in df['g'].unique():
    #    d[i] = [df['gen'][j] for j in df[df['g']==i].index]
    li_res.append(df.to_numpy().flatten())

li = np.concatenate((li_vom, li_gen), axis=1)
li_res = np.array(li_res)


#ic(len(li), li_gen)
#ic(len(li_res), li_res)

# print('li = ',li)
# print('li_res = ',li_res)
# frame = pd.concat(li, axis=0, ignore_index=True)
# print(frame)


test_ratio = 0.2
valid_ratio = 0.1


x_train, x_test, y_train, y_test = train_test_split(li, li_res, test_size=test_ratio, random_state=1, shuffle=True)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=1,
                                                      shuffle=True)


x_train = torch.Tensor(x_train).float()
x_test = torch.Tensor(x_test).float()
x_valid = torch.Tensor(x_valid).float()

y_train = torch.Tensor(y_train).float()
y_test = torch.Tensor(y_test).float()
y_valid = torch.Tensor(y_valid).float()

y = torch.Tensor(li_res)
x = torch.Tensor(li)

ic(li.shape, li_res.shape)

xmean = x.mean(dim=0)
ymean = y.mean(dim=0)
xstd = x.std(dim=0)
ystd = y.std(dim=0)

ic(xstd, ystd)

#x_train = transforms.Normalize(x_train, mean=xmean, std=xstd)
#x_test = transforms.Normalize(x_test, mean=xmean, std=xstd)
#x_valid = transforms.Normalize(x_valid, mean=xmean, std=xstd)

x_train = (x_train - xmean) / xstd
x_test = (x_test - xmean) / xstd
x_valid = (x_valid - xmean) / xstd

y_train = (y_train - ymean) / ystd
y_test = (y_test - ymean) / ystd
y_valid = (y_valid - ymean) / ystd

ic(np.isnan(x_train), np.isnan(y_train))

num_hidden1 = 10
num_features = x_train.shape[1]
num_classes = 1

model = nn.Sequential(nn.Linear(num_features, num_hidden1),
                      nn.ReLU(),
                      nn.Linear(num_hidden1, num_classes),
                      )

# model_ft = models.resnet152(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 120)
# Loss
loss = nn.MSELoss()
# Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train
num_epochs = 2000
num_samples_train = x_train.shape[0]
num_samples_valid = x_valid.shape[0]
num_samples_test = x_test.shape[0]

# x = torch.FloatTensor(x)
# yt = torch.LongTensor(yt)
# x = torch.tensor(x).float()
# yt = torch.tensor(yt).long()
i = []
l1 = []
l2 = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    yp = model(x_train)
    loss_value = loss(yp, y_train)
    loss_value.backward()
    optimizer.step()
    yp = model(x_valid)
    loss_value_valid = loss(yp, y_valid)
    writer.add_scalar("Loss/train", loss_value, epoch)
    print('Epoch: ', epoch, ', Train Loss: ', loss_value.item(),
          ', Validation Loss: ', loss_value_valid.item())
    i.append(epoch)

    l1.append(loss_value.data.numpy())
    l2.append(loss_value_valid.data.numpy())

writer.flush()

plt.figure(figsize=(10, 4))
plt.plot(i, l1, color="blue", label='train MSE')
plt.plot(i, l2, color='red', label='validation MSE')
plt.title('Cost Function vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

# yp = model(x_test)
# num_corrects = torch.sum(yp[:, 0].round() == y_test)
# acc_test = num_corrects.float() / float(num_samples_test)
# print('Test Accuracy: ', acc_test.item())
