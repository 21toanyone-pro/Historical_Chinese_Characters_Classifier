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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from net import Net
from torchvision.utils import save_image
import shutil
import natsort

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
# testset = torchvision.datasets.ImageFolder(root="./test/", transform= transform)
# classes = testset.classes
# testloader = DataLoader(testset, batch_size= 4, shuffle=False, num_workers=4)

path = 'D:/bri/'#구분할 이미지 파일 폴더 경로 입력(input) #S:/bri/
output_path = 'D:/clean/'#구분한 이미지파일 저장할 경로 입력(output)
pth_file = 'cifar_net.pth'# pth파일 디렉토리 직접입력(weight)

#===================================================================#

path_list = os.listdir(path)
path_list = natsort.natsorted(path_list)

listlist =[]
copylist = []
file_name =[]
ac = listlist.append #이미지 경로
copy = copylist.append
fileappend = file_name.append #이미지 이름

name = 'D:/clean/'

for i in path_list: # 왕 이름
    pathA = path + str(i) +'/'
    pathA_Copy = name + str(i) +'/'

    pathA_list = os.listdir(pathA)
    pathA_list = natsort.natsorted(pathA_list)
    for j in pathA_list: # 권 수
        pathB = pathA + str(j) +'/'
        pathB_C = pathA_Copy + str(j) +'/'

        pathB_list = os.listdir(pathB)
        pathB_list = natsort.natsorted(pathB_list)
        for k in pathB_list: # 장 수
            pathB_l = str(pathB)+str(k)
            pathB_l_C = str(pathB_C)+str(k)
            pathC_list = os.listdir(pathB_l)
            pathC_list = natsort.natsorted(pathC_list)
            for l in pathC_list:
                pathC_l = pathB_l+'/'+str(l) # 경로 + 이미지 파일 이름
                pathC_l_C = pathB_l_C
                ac(str(pathC_l)) #경로 저장
                copy(str(pathC_l_C))
                fileappend(l) #이름만 저장

i=0
if __name__ == '__main__':
    
    files = []
    net = Net() 
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    
    correct = 0
    total = 0

    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))

    for f in files:
        img = Image.open(f)  # Load image as PIL.Image
        x = transform(img)  # Preprocess image
        x = x.unsqueeze(0)  # Add batch dimension
        model = torch.load_state_dict(torch.load('./cifar_net_adam.pth'))
        output = model(x)  # Forward pass
        pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
            

        if not os.path.exists(output_path):
                os.makedirs(output_path)
            
        if pred == 0:
            if not os.path.isdir(copylist[i]):
                os.makedirs(copylist[i]+'/')
            shutil.copy( listlist[i], copylist[i])
            print("predicted as A01" )
        elif pred == 1:
            if not os.path.isdir(copylist[i]):
                os.makedirs(copylist[i]+'/')
            shutil.copy( listlist[i],'D:/A02/')
            print("predicted as A02" )
        i=i+1

