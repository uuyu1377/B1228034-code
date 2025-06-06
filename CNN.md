---
title: CNN

---

#  Dog Breed Identification
## ==CNN==
**檔案上傳與解壓縮**
```iphnb
from google.colab import files
uploaded = files.upload()
```
讓你手動從電腦上傳檔案到 Colab。
```ipynb
import zipfile
import os

zip_path = 'dog-breed-identification.zip'
extract_path = 'dog_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```
把剛剛上傳的壓縮檔解壓縮到 ```dog_data``` 資料夾裡。
```zip_path```: 上傳的壓縮檔。
```extract_path```: 指定解壓的目錄，資料會存在這裡。

**設定資料夾路徑**
```ipynb
base_dir = os.path.join(extract_path, 'dog-breed-identification')
train_dir = os.path.join(base_dir, 'train')
labels_path = os.path.join(base_dir, 'labels.csv')
sample_submission_path = os.path.join(base_dir, 'sample_submission.csv')
```
設定各種檔案的路徑。

**安裝函式庫**
```ipynb
!pip install seaborn pillow
```
安裝 ```seaborn```處理畫圖和 ```pillow```圖片處理用。

**資料初步分析與視覺化**
```ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import zipfile
```
載入必要的函式庫，```Pandas```用來分析資料、```Matplotlib/Seaborn```畫圖、```PIL```用來看圖片。
```ipynb
df = pd.read_csv(labels_path)
df['filename'] = df['id'] + '.jpg'
print(df.head())
```
```labels.csv``` 是標籤檔，裡面有兩欄：id和 breed（狗的品種）。
加一欄 ```filename```，因為實際圖片是用 id.jpg 命名的。
```ipynb
print("品種總數：", df['breed'].nunique())
print("\n前5名常見品種：\n", df['breed'].value_counts().head())
```
```nunique()```：統計不同品種的數量。
```value_counts()```：統計每種狗的數量並由高到低排序。
**畫圖（了解資料分佈）**
```ipynb
plt.figure(figsize=(12, 6))
top_breeds = df['breed'].value_counts()[:20]
sns.barplot(x=top_breeds.index, y=top_breeds.values)
plt.xticks(rotation=45)
plt.title("前20名常見的狗品種")
plt.xlabel("品種")
plt.ylabel("圖片數量")
plt.tight_layout()
plt.show()
```
用 ```Seaborn``` 畫出前 20 種狗的圖片數量分佈，幫助我們了解資料是否不平均，例如：某些品種照片特別多。

**建立訓練集與驗證集**
```ipynb
plt.figure(figsize=(12, 6))
top_breeds = df['breed'].value_counts()[:20]
sns.barplot(x=top_breeds.index, y=top_breeds.values)
plt.xticks(rotation=45)
plt.title("前20名常見的狗品種")
plt.xlabel("品種")
plt.ylabel("圖片數量")
plt.tight_layout()
plt.show()
```
圖片像素值標準化到 0~1，並設定訓練資料有 20% 拿去當驗證資料。
```ipynb
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='dog_data/train',
    x_col='filename',
    y_col='breed',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
```
**Keras 自動處理圖片與標籤的工具。**
→ 把圖片 resize 成 (150, 150)。
→ 每 32 張為一batch。
→ 把品種轉換成 one-hot encoding，因為是多類別分類。
→ 建立訓練用的資料集```subset='training'```。
```ipynb
valid_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory='dog_data/train',
    x_col='filename',
    y_col='breed',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)
```
驗證資料集的生成方式與訓練集一樣，只是 ```subset='validation'```，讓模型可以驗證表現。

**建立 CNN 模型**
```ipynb
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

num_classes = df['breed'].nunique()
```
匯入建立模型用的函式。
```num_classes``` ：總共有幾種狗，作為輸出層的神經元數量。
```ipynb
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```
CNN 結構。兩層卷積加池化，後面加 Dense 全連接層和 dropout，最後用 softmax 做多類別分類。

**編譯與訓練模型**
```ipynb
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
**告訴模型用哪種訓練方法與評估指標**：
優化器：adam。
loss：多分類要用 categorical_crossentropy。
```ipynb
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10
)
```
訓練 10 個 epoch，同時追蹤訓練與驗證集的表現。

**視覺化訓練過程**
```ipynb
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')

plt.tight_layout()
plt.show()
```
畫出 Loss 和 Accuracy 的變化圖，可以判斷模型是否過擬合

**測試集預測與輸出提交檔案**
```ipynb
sample_submission = pd.read_csv(sample_submission_path)
test_df = pd.DataFrame({'filename': sample_submission['id'] + '.jpg'})
```
建立一份測試資料表，準備進行預測。但測試集沒有標籤，只知道每張圖片的 id。
```ipynb
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col=None,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)
```
建立測試圖片的生成器，只會回傳圖片，不會回傳標籤。
```shuffle=False``` ：為了跟提交檔對齊順序。
```ipynb
predictions = model.predict(test_generator, verbose=1)
```
對測試資料進行預測，並輸出每張圖對應到每個品種的機率。
```ipynb
class_indices = train_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
breed_labels = [inv_class_indices[i] for i in range(len(inv_class_indices))]
```
把 ```class_indices```反轉，並變成 index → 品種名稱。
```breed_labels```：最後每一欄的標題。
```ipynb
submission = pd.DataFrame(predictions, columns=breed_labels)
submission.insert(0, 'id', sample_submission['id'])
submission.to_csv('submission.csv', index=False)
```
**把預測結果轉成符合 Kaggle 格式的 CSV。**
→ 第一欄是圖片 id，其餘是每種狗的機率。