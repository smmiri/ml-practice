from icecream import ic
import os
import pandas as pd
import copy
import time
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchtest import assert_vars_change


writer = SummaryWriter()

path = os.getcwd()  # use your path
ic(path)
os.chdir(path)

li_dem = []
li_gen = []
li_res = []
li_vom = []
for i in range(0, 1000):
    df = pd.read_csv('capacity_expansion/gen_data_{}.csv'.format(i), usecols=['cap'])
    li_gen.append(df.to_numpy().flatten())

    df = pd.read_csv('result_cap_{}.csv'.format(i), usecols=['cap'], skiprows=[1, 6, 7])
    li_res.append(df.to_numpy().flatten())

li = np.array(li_gen)
li_res = np.array(li_res)

data = li

#ic(len(li), li_gen)
#ic(len(li_res), li_res)

# print('li = ',li)
# print('li_res = ',li_res)
# frame = pd.concat(li, axis=0, ignore_index=True)
# print(frame)

device = 'cpu'

test_ratio = 0.2
valid_ratio = 0.1


x_train = data
y_train = li_res

batch_size = 16
output= y_train.shape[1]
num_features = x_train.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_ratio, random_state=1, shuffle=False)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=1, shuffle=False)

x_train = torch.tensor(x_train).float()
x_test = torch.tensor(x_test).float()
x_valid = torch.tensor(x_valid).float()

y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()
y_valid = torch.tensor(y_valid).float()


y = torch.Tensor(li_res)
x = torch.Tensor(li)

xmean = x.mean(dim=0)
ymean = y.mean(dim=0)
xstd = x.std(dim=0)
ystd = y.std(dim=0)

#x_train = transforms.Normalize(x_train, mean=xmean, std=xstd)
#x_test = transforms.Normalize(x_test, mean=xmean, std=xstd)
#x_valid = transforms.Normalize(x_valid, mean=xmean, std=xstd)

x_train = (x_train - xmean) / xstd
x_test = (x_test - xmean) / xstd
x_valid = (x_valid - xmean) / xstd

y_train = (y_train - ymean) / ystd
y_test = (y_test - ymean) / ystd
y_valid = (y_valid - ymean) / ystd

train_dataset = TensorDataset(x_train,y_train)
valid_dataset = TensorDataset(x_valid,y_valid)
test_dataset = TensorDataset(x_test,y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

one_batch_train = next(iter(train_loader))

class res_block_1d(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU):
        super(res_block_1d, self).__init__()
        self.activation = activation()
        self.block = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Linear(in_dim, out_dim),
            activation(),
            nn.BatchNorm1d(out_dim),
            nn.Linear(out_dim, out_dim),
            activation(),
        )

    def forward(self, x):
        out = self.activation(self.block(x) + x)

        return out

class resnet(nn.Module):
    def __init__(self, output, activation=nn.ReLU):
        super(resnet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            #res_block_1d(128, 128),
            #nn.Dropout(p=0.5),
            BatchNorm1d(128),
            activation(),
            nn.Linear(128, 128),
            #res_block_1d(128, 128),
            #nn.Dropout(p=0.5),
            BatchNorm1d(128),
            activation(),
            nn.Linear(128, 128),
            #res_block_1d(128, 128),
            #nn.Dropout(p=0.5),
            activation(),
            nn.Linear(128, 128),
            #res_block_1d(128, 128),
            #nn.Dropout(p=0.5),
            BatchNorm1d(128),
            activation(),
            nn.Linear(128, 128),
            #res_block_1d(128, 128),
            #nn.Dropout(p=0.5),
            BatchNorm1d(128),
            activation(),
            #res_block_1d(128, 128),
            nn.Linear(128, output)

        )

    def forward(self, x):
        out = self.net(x)

        return out

model = resnet(output).to(device)
print(model)


# criterion = nn.CrossEntropyLoss()
loss = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, weight_decay=0.001)

dataloaders_dict = {'train': train_loader, 'val': valid_loader}
num_epochs = 1000

writer = SummaryWriter()

def train_model(model, dataloaders, loss, optimizer, num_epochs=num_epochs):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, y_train in dataloaders[phase]:
                inputs = inputs.to(device)
                y_train = y_train.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)

                    loss_value = loss(outputs, y_train)
                    # print('a=',loss_value)
                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == y_train.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)
                writer.add_scalar('Loss/test', epoch_loss, epoch)
            else:
                train_acc_history.append(epoch_loss)
                writer.add_scalar('Loss/train', epoch_loss, epoch)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(0))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

model, val_hist, train_hist = train_model(model, dataloaders_dict, loss, optimizer, num_epochs=num_epochs)

# # Plot the training curves of validation accuracy vs. number
# #  of training epochs


plt.title("Test Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),val_hist,label="Pretrained")
plt.xticks(np.arange(0, num_epochs+1, 100.0))
plt.legend()
plt.show()


plt.title("Train Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Accuracy")
plt.plot(range(1,num_epochs+1),train_hist,label="Pretrained")
plt.xticks(np.arange(0, num_epochs+1, 100.0))
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(range(1,num_epochs+1),val_hist, color = "blue",label = "Validation")
plt.plot(range(1,num_epochs+1),train_hist, color = 'red',label = "Training")
plt.title("Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel('MSE')
plt.xticks(np.arange(0, num_epochs+1, 100.0))
plt.legend()
plt.show()