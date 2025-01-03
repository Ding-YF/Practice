import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
# 创建数据集
class Img_Dataset(Dataset):
    '''
    imput: images path 
    output: images list; PIL/tensor type
    '''
    # 初始化传入的参数， 
    def __init__(self,img_dir,crop_zise):
        self.img_dir = img_dir
        self.img_list = os.listdir(self.img_dir)
        self.compose = transforms.Compose([
            transforms.CenterCrop(crop_zise),
            transforms.ToTensor()
        ])
        self.crop_size = crop_zise
    def __len__(self):
        return len(self.img_list)
    # 数据集实例化以后可以直接调用，如 dataset(10)
    def __getitem__(self,idx):
        img_name = self.img_list[idx] 
        img_path = os.path.join(self.img_dir,img_name)
        img = Image.open(img_path)
        img_tensor = self.compose(img)
        '''
        Note: convert img to RGB
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                print(f"Converting image to RGB: {img_path}, original mode: {img.mode}")
                img = img.convert('RGB') # 强制转换为RGB

        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image: {img_path}, Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading image: {img_path}, Error: {e}")
            return None
        
        img_tensor = self.transforms(img)
        '''
        return img_tensor

# 图片卷积
class Base_Model(nn.Module):
    def __init__ (self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5,stride=1,padding=0)
        # self.conv2 = nn.Conv2d(20,20,5)
    def forward(self,input):
        output = F.relu(self.conv1(input))    
        # x_2 = F.relu(self.conv2(output))    
        return output

# 图片池化
class Base_pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(5,stride=5,ceil_mode=True)
    def forward(self,input):
        output = self.pool(input)
        return output

# Simgmoid
class Base_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self,input):
        output = self.sigmoid(input)
        return output
# 线性化 
class Base_linear(nn.Module):
    def __init__(self,features_in,features_out):
        super().__init__()
        self.linear1 = nn.Linear(features_in,features_out)
    def forward(self,input):
        output = self.linear1(input)
        return output

# CIFAR 10 Processing
class CR_P(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )
    def forward(self,input):
       output = self.sequential(input) 
       return output

# Tensorboard 可视化
def board(title,img,idx):
    '''
    imput: title,images tensor type/numpy type,number of list 
    output: None
    '''
    writer = SummaryWriter("logs")
    writer.add_images(f"{title}",img,idx)
    writer.close()

# Image type:PIL -> tensor
def transform(img):
    '''
    imput: images PIL type 
    output: images tensor type
    '''
    totensor = transforms.ToTensor()
    img_tensor = totensor(img)
    return img_tensor

# Normlize 图片归一化
def normlize(img):
    '''
    imput: images tensor type 
    output: normlized images
    '''
    tonorm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    img_norm = tonorm(img)
    return img_norm

def trans_randomCrop(target_img):
    '''
    function: random crop image and write in tensorboard
    imput: image type PIL
    '''
    randomCrop = transforms.RandomCrop(600,600)
    trans_compose = transforms.Compose([randomCrop,transform])
    for i in range(50):
        img_crop = trans_compose(target_img)
        board("randomCrop_2",img_crop,i)        

# Online Dataset
trans_ol = transforms.Compose([
    transforms.ToTensor()
])
datasets_train = datasets.CIFAR10(
    "datasets",
    train=True,
    transform=trans_ol,
    download=False
)
datasets_test = datasets.CIFAR10(
    "datasets",
    train=False,
    transform=trans_ol,
    download=False
)

# datasets loader
img_dir = "/home/Echo/code/dataset/CLoT_cn_2000/ex_images"
img_dataset = Img_Dataset(img_dir,crop_zise=450)
data_loader_finetuning = DataLoader(img_dataset,batch_size = 64,shuffle = False,num_workers = 0,drop_last = False)
test_data_loader = DataLoader(dataset = datasets_test,batch_size = 64,shuffle = False,num_workers = 0,drop_last = False)
train_data_loader = DataLoader(dataset = datasets_train,batch_size = 64,shuffle = False,num_workers = 0,drop_last = False)

if __name__ == '__main__':
    # 初始参数
    '''
    Note:
        PIL: type(img)/img.size
        tensor: img.shape
    '''
    # 卷积
    '''
    Rnn = Base_Model()
    step = 1 
    for data in data_loader:
        imgs,targets = data
        img_rnn = Rnn(imgs)
        img_rnn = torch.reshape(img_rnn,(-1,3,28,28))
        board("Rnn",img_rnn,step)
        board("origin",imgs,step)
        step+=1
    '''
    #池化,减小数据量
    '''
    pool = Base_pool()
    step = 1
    for data in data_loader:
        if step<=10:
            imgs,tragets = data
            img_pooled = pool(imgs)
            board("Pool2",img_pooled,step)
            step+=1
    '''
    # Sigmoid
    '''
    sigmoid = Base_sigmoid()
    step = 1
    for data in data_loader:
        if step <=10:
            imgs,targets = data
            img_sigmoid = sigmoid(imgs)
            board("sigmoid",img_sigmoid,step)
            step+=1
    '''
    # Dataset img Shape
    '''
    img_ex = next(iter(data_loader))
    imgs,tragets = img_ex
    # print(imgs[2].shape[1])

    # linear
    linear = Base_linear(imgs[2].shape[1],10)
    step = 0
    for data in data_loader:
        if step <=10:
            imgs,targets = data
            img_linear = linear(imgs)
            print(img_linear.shape) 
            board("linear",img_linear,step)
            step+=1
    '''
    # CIFAR 10 Processing
    '''
    cr_p = CR_P()
    correct = 0
    step = 0
    for data in data_loader:
        if step<1:
            imgs,targets = data
            outputs = cr_p(imgs)
            _, predicted = torch.max(outputs.data, 1)
            step += targets.size(0)
            correct += (predicted == targets).sum().item()
            print(targets)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / step))
    '''
    # Optimize
    cr_p = CR_P()
    if torch.cuda.is_available:
        cr_p = cr_p.cuda()
    epoch = 30
    learn_loss = 1e-2
    train_step = 0
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available:
        loss = loss.cuda()
    optimizer = torch.optim.SGD(cr_p.parameters(),lr=learn_loss)
    writer = SummaryWriter("logs")
    for i in range(epoch):
        start_time = time.time()
        print(f"===Training in epoch {i+1}===")
        total_loss = 0
        # train in each epoch
        for data in train_data_loader:
            optimizer.zero_grad()
            imgs,targets = data
            if torch.cuda.is_available:
                imgs = imgs.cuda()
                targets = targets.cuda()
            train_outputs = cr_p(imgs)
            train_output_loss = loss(train_outputs,targets)
            train_output_loss.backward()
            optimizer.step()
            total_loss += train_output_loss.item()
            train_step+=1
            if train_step % 100 == 0:
                endtime = time.time()
                print(f"Setp 100 duration: {endtime-start_time}s")
                print(f"Training_step:{train_step} Loss:{train_output_loss}")
                writer.add_scalar("SGD_OPTIMIZE",train_output_loss.item(),train_step)
                writer.close()
        # test in each epoch
        with torch.no_grad():
            accracy = 0
            total_accracy = 0
            test_step = 0
            test_data_length = len(datasets_test)
            for data in test_data_loader:
                imgs,targets = data
                if torch.cuda.is_available:
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                test_outputs = cr_p(imgs)
                accracy = (test_outputs.argmax(1) == targets).sum()
                total_accracy += accracy
            print(f'Accuracy of the network on the epoch {i+1} is {(total_accracy/test_data_length)*100}%')
        save_path = os.path.join("./step",f"CIFAR-10_Model_epoch{i}.pth")
        torch.save(cr_p,save_path)
    # 调试
