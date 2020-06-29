from PIL import Image
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import os
import cv2
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from net import Net
from torchvision.utils import save_image

trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

trainset = torchvision.datasets.ImageFolder(root="./img/", transform= trans)
classes = trainset.classes
trainsloader = DataLoader(trainset, batch_size= 16, shuffle=True, num_workers=4)


if __name__ == '__main__':
    
    net = Net() 
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion = torch.nn.DataParallel(criterion)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(50):   # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(trainsloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()
            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)