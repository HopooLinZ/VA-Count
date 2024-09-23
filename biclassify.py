import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import clip
import re
import torchvision.models as models
# 1. 读取数据和预处理
def read_label_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            image_name, label = line.strip().split(',')
            data.append([image_name, 1 if label == 'one' else 0])
    return pd.DataFrame(data, columns=['image', 'label'])
# 读取a.txt中的图片名称
with open('./data/FSC147/train.txt', 'r') as file:
    a_txt_images = file.read().splitlines()

# 提取.jpg前的数字
a_txt_numbers = set([name.split('.')[0] for name in a_txt_images])

# 从label.txt中读取图片名称和标签
with open('./data/FSC147/one/labels.txt', 'r') as file:
    label_txt_lines = file.read().splitlines()

# 筛选出存在于a.txt中的图片
filtered_images = []
for line in label_txt_lines:
    image_name, label = line.strip().split(',')
    # 使用正则表达式匹配开头的数字
    match = re.match(r'(\d+)', image_name)
    if match:
        image_number = match.group(1)
        if image_number in a_txt_numbers:
            # 转换'label'的值
            label_value = 1 if label == 'one' else 0
            filtered_images.append([image_name, label_value])  # 注意这里是列表，以匹配read_label_file的输出

# 将筛选后的图片和标签转换为DataFrame，确保列名与read_label_file函数的输出相匹配
df_filtered = pd.DataFrame(filtered_images, columns=['image', 'label'])

# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. 数据集划分
data_folder = './data/FSC147/one'
label_file = os.path.join(data_folder, 'labels.txt')
# df = read_label_file(label_file)
df = df_filtered
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. 数据加载
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

train_dataset = CustomDataset(train_df, data_folder, transform=transform)
test_dataset = CustomDataset(test_df, data_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. 模型定义
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super(ClipClassifier, self).__init__()
        self.clip_model = clip_model
        # 冻结CLIP模型的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)  # 二分类

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
        x = self.fc(image_features)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        # 加载预训练的ResNet50模型
        self.resnet50 = models.resnet50(pretrained=True)
        # 冻结所有预训练层的参数
        for param in self.resnet50.parameters():
            param.requires_grad = False
        # 替换最后的全连接层以适应二分类任务
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, images):
        return self.resnet50(images)

# 5. 训练和测试
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
clip_model, _ = clip.load("ViT-B/32", device=device)
# model = ClipClassifier(clip_model).to(device)
model = ResNetClassifier().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return 100. * correct / len(test_loader.dataset)

best_accuracy = 0.0
for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), './data/out/classify2/best_model.pth')
        print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

