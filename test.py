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
from torchvision import models
#from torchsummary import summary
from torchvision import datasets, models, transforms

path = './test/'#구분할 이미지 파일 폴더 경로 입력(input) #S:/bri/
output_path = './class/'#구분한 이미지파일 저장할 경로 입력(output)

text = open('./recogo.txt', 'w', encoding='UTF8' )


#클래스 분류
def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

#===================================================================#

path_list = os.listdir(path)
path_list = natsort.natsorted(path_list)

listlist =[]
copylist = []
file_name =[]
ac = listlist.append #이미지 경로
copy = copylist.append
fileappend = file_name.append #이미지 이름

name = './class/'

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


files = []

# r=root, d=directories, f = files           
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))
dir2 = './img/train/'
class_name, class_to_idx = find_classes(dir2)

for target_class in sorted(class_to_idx.keys()):
    class_index = class_to_idx[target_class]

for f in files:
    img = Image.open(f).convert("RGB")  # Load image as PIL.Image
    x = transform(img)  # Preprocess image
    x = x.unsqueeze(0)  # Add batch dimension
    model = torchvision.models.resnet152(num_classes = 186) # 모델 
    model.load_state_dict(torch.load('cifar_ResNet1818.pth'))
    model.eval()
    output = model(x)  # Forward pass
    pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
        
    if not os.path.exists(output_path):
            os.makedirs(output_path)

    
    for i in range(0,len(class_name)):
        if pred == i:
            if not os.path.exists(output_path+'/'+str(class_name[i])):
                os.mkdir(output_path+'/'+str(class_name[i]))
            shutil.copy( f ,output_path+'/'+str(class_name[i]))
            print("predicted as " +str(class_name[i]))
            text.write(str(class_name[i]))



