---
title: EfficientNet-B0

---

#  Dog Breed Identification
## ==EfficientNetB0==
**檔案上傳與解壓縮**
```iphnb
from google.colab import files
uploaded = files.upload()
```
讓你手動從電腦上傳檔案到 Colab。
```iphnb
import zipfile
zip_path = "dog-breed-identification.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("dog-breed-identification")
```
解壓縮檔案，把內容放到 ```dog-breed-identification``` 資料夾中。
```iphnb
data_dir = "dog-breed-identification"
print(os.listdir(data_dir))
```
列出解壓後的資料夾內容，確認裡面有 ```train```, ```test```,```labels.csv``` 等資料。

**套件與資料前處理**
```iphnb
import os, pandas as pd, numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
```
引入必要的函式庫，```os & PIL```用來處理圖片檔案，```pandas```讀取與處理標籤資料，```sklearn```用來切分訓練/驗證集，還有計算模型評估指標。
```iphnb
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
```
PyTorch 系統套件，處理訓練資料、自定義模型、優化器等。
```iphnb
import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
```matplotlib```、```seaborn``` 進行畫圖分析，並使用 GPU，否則用 CPU。
```iphnb
labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
```
讀取訓練標籤檔
```iphnb
breed_labels = labels_df['breed'].unique()
breed_to_idx = {breed: idx for idx, breed in enumerate(breed_labels)}
labels_df['label'] = labels_df['breed'].map(breed_to_idx)
```
將狗的品種轉為「整數編號」：
例如：
"golden_retriever" → 0
"poodle" → 1
"boxer" → 2
這樣模型才能進行分類任務。

**資料集切分**
```iphnb
train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label'], random_state=42)
```
把資料分為訓練集（80%）與驗證集（20%），並使用 ```stratify``` 確保每個品種的分布比例平均，避免偏差。

**自定義 Dataset 類別**
```iphnb
class DogBreedDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id']
        label = self.dataframe.iloc[idx]['label']
        img_path = os.path.join(self.data_dir, "train", f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
```
PyTorch 的自定義資料集類別：
```__len__```：告訴你這個資料集有幾張圖片。
```__getitem__```：第幾筆資料怎麼取出：從 train/ 資料夾找圖片，載入 + 轉成 RGB，再進行 transform。
最後回傳 ```(image_tensor, label_int)```。

**設定圖像處理方式（transforms）**
```iphnb
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```
資料增強：
→ 隨機裁切 + 水平翻轉 + 色彩擾動
→ 正規化成與 ImageNet 一樣的平均值、標準差
```iphnb
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```
驗證集使用固定處理流程，避免評估結果不穩。

**建立 DataLoader**
```iphnb
train_dataset = DogBreedDataset(train_df, data_dir, transform=train_transform)
val_dataset = DogBreedDataset(val_df, data_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```
把 Dataset 用進 DataLoader：每次抓一批資料（batch）加快訓練速度
```shuffle=True``` ：打亂訓練順序有助模型學習

**模型建構與微調**
```iphnb
model = models.efficientnet_b0(pretrained=True)
```
載入 EfficientNet-B0 模型
```iphnb
for param in model.parameters():
    param.requires_grad = True
```
允許所有層更新權重
```iphnb
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(breed_labels))
```
修改最後一層，把預設 1000 類分類器換成 120 類。

**定義損失函數與優化器**
```iphnb
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```
優化器：Adam 。
損失函數：分類用的 Cross-Entropy。→ 每 5 個 epoch 學習率衰減為一半。

**模型訓練迴圈**
```iphnb
for epoch in range(20):
    model.train()
    running_loss = 0.0
```
**模型訓練階段**
```model.train()```：PyTorch 的模式切換，會啟用像 Dropout 等訓練用的機制。
```running_loss``` ：計算整個 epoch 的平均訓練損失。
```iphnb
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
```
**每個 batch 的訓練過程**
```images, labels = images.to(device)```：從資料集中一個個batch抓出來，送到 GPU 上。
```optimizer.zero_grad()```：每次訓練要清空之前的梯度，不然會累加。
```outputs = model(images)```：模型對這批圖片做預測。
```loss = criterion(outputs, labels)```：用Cross-Entropy計算這批預測錯多少，也就是loss。
```loss.backward()```：反向傳播，算出梯度。
```optimizer.step()```：更新模型的參數。
```running_loss += loss.item()```：把這批的 loss 加總起來，之後平均。
```iphnb
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
```
**計算並記錄這一輪的平均訓練損失**
把所有 batch 的 loss 平均起來，存在 ```train_losses``` 中

**計算訓練集準確率**
```iphnb
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
```
```model.eval()```：切到驗證的模式，並關掉 dropout 等機制。
```torch.no_grad()```：禁用梯度計算，加快推論速度、節省記憶體。
```iphnb
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
```
**判斷是否預測正確**
```torch.max(outputs, 1)```：取每筆預測的最大值對應的類別。
```(predicted == labels)```：比對預測與正確答案。
```.sum().item()```：計算正確的筆數。
```iphnb
    train_acc = train_correct / train_total
    train_accuracies.append(train_acc)
```
 把這一輪訓練集的準確率記下來。
 
 **驗證階段**
 ```iphnb
    correct = 0
    total = 0
    val_running_loss = 0.0
    epoch_true = []
    epoch_pred = []
```
準備紀錄驗證損失、準確率，還有所有真實標籤與預測值。
```iphnb
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
```
用和上面相同的方式：
→ 推論圖片分類
→ 累加預測的loss
→ 記下正確數量 & 預測值
```iphnb
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_true.extend(labels.cpu().numpy())
            epoch_pred.extend(predicted.cpu().numpy())
```
**儲存驗證損失與準確率**
```iphnb
    val_loss = val_running_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
```
跟訓練集一樣，平均損失、準確率，存起來畫圖用。

**儲存最佳模型**
```iphnb
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"儲存最佳模型（val_acc={val_acc:.4f}）")
```
若目前驗證準確率比歷來最好還高 → 儲存這個 epoch 的模型為 ```best_model.pth```。

**學習率調整**
```iphnb
    scheduler.step()
```
使用學習率排程器，每 5 個 epoch 自動降低學習率。
```iphnb
    print(f"Epoch {epoch+1}/20 - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```
印出這一輪訓練的結果。

**儲存每一輪的預測與真實標籤**
```iphnb
    y_true = epoch_true
    y_pred = epoch_pred
```
最後把整個 epoch 的預測與真實答案存起來。

**畫訓練過程曲線**
```iphnb
plt.figure(figsize=(12, 5))

# Loss 圖
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy 圖
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
```
觀察模型是否 overfitting、是否學習穩定。

**處理測試資料與預測**
```iphnb
test_dir = os.path.join(data_dir, "test")
test_files = os.listdir(test_dir)
test_files = [f for f in test_files if f.endswith(".jpg")]
test_ids = [os.path.splitext(f)[0] for f in test_files]
```
載入 test/ 中的所有圖片，準備做預測。

**自訂測試集 Dataset**
```iphnb
class TestDataset(Dataset):
    def __init__(self, file_list, img_dir, transform=None):
        self.file_list = file_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_id = self.file_list[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_id

test_dataset = TestDataset(test_ids, test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
類似訓練集 Dataset，但不需要標籤，回傳圖片 + 圖片 ID。

**載入最佳模型並產生預測檔**
```iphnb
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```
載入最好的模型權重進行推論。
```iphnb
with torch.no_grad():
```
不要計算梯度，因為在測試階段不需要反向傳播和更新參數，可以省下很多記憶體與運算資源。
```iphnb
for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)  # shape: (batch, 120)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  
        for img_id, prob in zip(ids, probs):
            row = [img_id] + prob.tolist()
            submission.append(row)
```
從 ```test_loader``` 抓出一圖片與 id，並會保留原始 image_id 來對應預測結果。
把圖片傳到 GPU，並送進模型，輸出 logits。
把 logits 轉換成softmax，每一行的加總會是 1。
```dim=1``` ：對每個樣本做 softmax。
```.cpu().numpy()``` ：把 tensor 搬回 CPU 並轉成 NumPy 陣列，方便之後處理或存成 CSV。
最後把每張圖片的id 和對應的 120 個類別機率組成一列
```iphnb
submission_df.to_csv("EfficientNetB0submission.csv", index=False)
files.download("EfficientNetB0submission.csv")
```
將預測結果存成符合 Kaggle 格式的 CSV：每一列是狗的 ID + 每個品種的機率。