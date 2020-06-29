from __future__ import division
import torch
torch.manual_seed(0)
from net import Net
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageOps, ImageEnhance
import torch.backends.cudnn as cudnn
import os
import shutil

path = './test_data'#구분할 이미지 파일 폴더 경로 입력(input)
output_path = './class'#구분한 이미지파일 저장할 경로 입력(output)
pth_file = 'cifar_net.pth'# pth파일 디렉토리 직접입력(weight)

# model = torch.load(pth_file) #gpu 있을때
# #model=torch.load(pth_file, map_location='cpu') #cpu만 있을때
# model = model.load_state_dict(torch.load(pth_file))

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])


files = []
if __name__ == '__main__':
    model = Net() 
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = False

    model.load_state_dict(torch.load('./cifar_net.pth'))
          
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))


    for f in files:
        print(f)
        img = Image.open(f)  # Load image as PIL.Image
        x = transform(img)  # Preprocess image
        x = x.unsqueeze(0)  # Add batch dimension

        output = model(x)  # Forward pass
        pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
            

        if not os.path.exists(output_path):
                os.makedirs(output_path)
            

        if pred == 0:
            if not os.path.exists(output_path+'/A01'):
                os.mkdir(output_path+'/A01')
            shutil.copy( f ,output_path+'/A01')
            print("predicted as A01" )
        elif pred == 1:
            if not os.path.exists(output_path+'/A02'):
                os.mkdir(output_path+'/A02')
            shutil.copy( f ,output_path+'/A02')
            print("predicted as A02" )
        elif pred == 2:
            if not os.path.exists(output_path+'/A03'):
                os.mkdir(output_path+'/A03')
            shutil.copy( f ,output_path+'/A03')
            print("predicted as A03" )

        elif pred == 3:
            if not os.path.exists(output_path+'/A04'):
                os.mkdir(output_path+'/A04')
            shutil.copy( f ,output_path+'/A04')
            print("predicted as A04" )
        
