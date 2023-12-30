import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from unet import *
from dataset import *
from tqdm import tqdm

EPOCHS = 200
lr = 1e-4
batch_size = 4
dim = (1451, 301)
TRAINING_SET_RATE = 0.8
htimgPath = '../data/htimg/'
htlblPath = '../data/htlbl/'
picPath = '../picture/'
modPath = '../model/'
if not os.path.exists(picPath):
    os.mkdir(picPath)
if not os.path.exists(modPath):
    os.mkdir(modPath)
# DEVICE = torch.device(torch.device('cuda') if torch.cuda.is_available() else 'cpu') # Nvidia GPU
DEVICE = torch.device(torch.device('mps') if torch.backends.mps.is_available() else 'cpu') # Apple GPU
print(DEVICE)


def go():
    load_data()
    epochs = EPOCHS
    model = UNet(in_ch=1, out_ch=1, size=dim).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-8)
    train_model(model, criterion, optimizer, scheduler, epochs)


def load_data():
    allID = []
    for fileName in os.listdir(htimgPath):
        if fileName.endswith('.dat'):
            allID.append(fileName)
    allLength = len(allID)
    trainLength = int(allLength * TRAINING_SET_RATE)
    trainID = allID[:trainLength]
    validID = allID[trainLength:]
    global trainData
    global validData
    trainData = DataLoader(DASDataset(htimgPath, htlblPath, trainID, chann=1, dim=dim), batch_size=batch_size, shuffle=True)
    validData = DataLoader(DASDataset(htimgPath, htlblPath, validID, chann=1, dim=dim), batch_size=batch_size, shuffle=False)
    print('data loaded')
    return trainID[0], validID[0]


def train_model(model, criterion, optimizer, scheduler, epochs):

    def plot(title, xlabel, ylabel, picFile, curve1, curve2=None, legend=None):
        plt.figure(figsize=(50, 30))
        plt.title(title, fontsize=48)
        plt.plot(curve1)
        plt.xlabel(xlabel, fontsize=40)
        plt.ylabel(ylabel, fontsize=40)
        if curve2:
            plt.plot(curve2)
            plt.legend(legend, loc='center right', fontsize=40)
        plt.tick_params(axis='both', which='major', labelsize=36)
        plt.tick_params(axis='both', which='minor', labelsize=36)
        plt.savefig(picFile)
        plt.close()

    MinTrainLoss = 1e5
    MinValidLoss = 1e5
    train_loss = []
    valid_loss = []
    lr_list = []
    print('ready to train')
    
    for epoch in range(epochs):
        # train
        total_train_loss = []
        model.train()
        for (input, label) in tqdm(trainData):
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())
        running_train_loss = np.mean(total_train_loss)
        running_lr = optimizer.param_groups[0]['lr']
        train_loss.append(running_train_loss)
        lr_list.append(running_lr)
        scheduler.step(running_train_loss)
        if train_loss[-1] < MinTrainLoss:
            file_path = modPath + '/ht_model_min_train.pth'
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, file_path)
            MinTrainLoss = train_loss[-1]

        # valid
        total_valid_loss = []
        model.eval()
        for (input, label) in tqdm(validData):
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, label)
            total_valid_loss.append(loss.item())
        running_valid_loss = np.mean(total_valid_loss)
        valid_loss.append(running_valid_loss)
        if valid_loss[-1] < MinValidLoss:
            file_path = modPath + '/ht_model_min_valid.pth'
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, file_path)
            MinValidLoss = valid_loss[-1]
        
        # print training log
        print(f'epoch{epoch+1:3d}/{epochs:3d}, train loss = {running_train_loss:6.4f}, valid loss = {running_valid_loss:6.4f}, lr = {running_lr}')

        # save model
        if (epoch+1)%10 == 0:
            file_path = modPath + '/ht_model%03d.pth'%(epoch+1)
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, file_path)
    
    # plot
    picFile1 = picPath + 'ht_loss.png'
    picFile2 = picPath + 'ht_learning_rate.png'
    plot(title='HT Loss During Training', curve1=train_loss, curve2=valid_loss, xlabel='Epoch', ylabel='Loss', legend=['train', 'valid'], picFile=picFile1)
    plot(title='HT Learning Rate During Training', curve1=lr_list, xlabel='Epoch', ylabel='Learning Rate', picFile=picFile2)
    print('training finished')



if __name__ == '__main__':
    go()