from PIL import Image
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import cv2

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from net import Net # 개인 커스텀 네트워크를 쓰고 싶을때

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import time
import copy
import natsort
import shutil
from adamp import AdamP

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=20, scale=(0.8, 1.3)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.4), ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomErasing()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

data_dir = "./img/"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=400,shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    model_conv = torchvision.models.resnet152(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 186)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer_ft = optim.Adam(model_conv.parameters(), lr=0.01) #lr=0.001, momentum=0.9
    optimizer_ft = AdamP(model_conv.parameters(), lr=0.01) #lr=0.001, momentum=0.9
    # 7 에폭마다 0.1씩 학습율 감소
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # net = Net() 
    # net = net.cuda()
    # net = torch.nn.DataParallel(net)
    # cudnn.benchmark = False  
    train_model(model_conv, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)

    PATH = './test0620.pth'   
    torch.save(model_conv.state_dict(), PATH)
