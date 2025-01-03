import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from PIL import Image

from indxDataset import CR_P

# Online Dataset
datasets_test = datasets.CIFAR10(
    "datasets",
    train=False,
    transform=transforms.ToTensor(),
    download=False
)
test_data_loader = DataLoader(dataset=datasets_test,batch_size=64,shuffle=False,num_workers=0,drop_last=False)
# Local Dataset
single_image_path = "./datasets/dog.png"
single_image = Image.open(single_image_path)
transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor()
])
single_image = transform(single_image)
single_image = torch.reshape(single_image,[1,3,32,32])
single_image_cuda = single_image.cuda()
# print(single_image.shape)

# Test Model in single image
C_model_trained = torch.load("CIFAR-10_Model_epoch18.pth")
C_model_trained.eval()
with torch.no_grad():
    single_image_output = C_model_trained(single_image_cuda)
print(single_image_output.argmax(1))

# Test Model in datasets
'''
C_model_trained = torch.load("CIFAR-10_Model_epoch20.pth")
totally_accuracy = 0
for data in test_data_loader:
    imgs,targets = data
    if torch.cuda.is_available:
        imgs = imgs.cuda()
        targets = targets.cuda()
    test_output = C_model_trained(imgs)
    accuracy = (test_output.argmax(1)==targets).sum()
    totally_accuracy += accuracy
print(f"Accuracy: {totally_accuracy/len(datasets_test)*100}%")
'''