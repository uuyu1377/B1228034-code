---
title: MobileNetV2_方法一

---

#  Dog Breed Identification
## ==MobileNetV2_方法一==
**掛載與解壓縮**
```iphnb
from google.colab import files
uploaded = files.upload() 
```
讓你從自己的電腦上傳解壓縮檔案到 Colab 裡。
```ipynb
import zipfile, os
zip_path = 'dog-breed-identification.zip'
extract_path = 'dog_data'
```
匯入 zip 檔解壓需要的模組，並設定變數。
```zip_path``` 是壓縮檔的檔名。
```extract_path``` 是我們要解壓縮到哪個資料夾。
```ipynb
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```
把壓縮檔解壓到 ```dog_data```資料夾裡，裡面會有圖片、labels.csv 等資料。

**安裝套件**
```ipynb
!pip install seaborn pillow
```
安裝額外的套件，```seaborn``` 用來畫圖，```pillow``` 是處理圖片的工具。

**匯入常用套件**
```ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
```pandas``` 處理表格資料，```matplotlib.pyplot``` 和 ```seaborn``` 用來畫圖分析。
```ipynb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
```
```ImageDataGenerator``` 幫我們處理圖片格式跟增強。
```MobileNetV2``` 是我們這次用的模型。
```preprocess_input``` 是這個模型對應的圖片前處理方法。
 匯入 Keras 建模型會用到的 layer，optimizer 還有 EarlyStopping。
```EarlyStopping``` ==**原理**：訓練誤差會一直下降，一開始驗證集誤差也會跟著下降，而當驗證集的誤差開始上升，就代表發生了過擬合==

**看資料夾裡面有什麼**
```ipynb
!ls dog_data
!ls dog_data/dog-breed-identification
```
 列出 ```dog_data``` 資料夾裡的內容，確認資料有解壓成功。
 
 **資料讀取與處理**
```ipynb
base_dir = 'dog_data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
labels_path = os.path.join(base_dir, 'labels.csv')

df = pd.read_csv(labels_path)
df['filename'] = df['id'] + '.jpg'
```
設定訓練資料、測試資料、標籤檔案的路徑。
用 pandas 把 labels.csv 讀進來，並新增一欄 ==filename==，是圖片的檔名。**（因為圖片用 ID 命名，後面要加上 .jpg 才找得到檔案）**

**看品種的分布情形**
```ipynb
print("品種總數：", df['breed'].nunique())
print("\n 前 5 名常見品種：\n", df['breed'].value_counts().head())
plt.figure(figsize=(12, 6))
top_breeds = df['breed'].value_counts()[:20]
sns.barplot(x=top_breeds.index, y=top_breeds.values)
plt.xticks(rotation=45)
plt.title("前 20 名常見的狗品種")
plt.xlabel("品種")
plt.ylabel("圖片數量")
plt.tight_layout()
plt.show()
```
把前 20 名品種畫成長條圖，觀察哪些狗出現得比較多。

**設定資料產生器**
```ipynb
img_size = 224
batch_size = 32
```
 圖片大小設 224x224，跟 MobileNetV2 規格一樣。一次訓練拿 32 張圖。
 ```ipynb
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
```
建立一個圖片資料產生器，會幫圖片做 MobileNetV2 專用的前處理，並自動切出 20% 當驗證集。
 ```ipynb
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_dir,
    x_col='filename',
    y_col='breed',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
```
設定訓練資料產生器，會讀進圖檔和對應的品種標籤，並做分類用的 one-hot 編碼。
 ```ipynb
valid_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_dir,
    x_col='filename',
    y_col='breed',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)
```
跟上面一樣，不過這是驗證集的版本。

**建立 MobileNetV2 模型**
 ```ipynb
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=Input(shape=(img_size, img_size, 3))
)
base_model.trainable = False
```
叫出 MobileNetV2 預訓練模型，不包含最後一層（因為我們要自己加），先不讓它更新參數（凍結）。
 ```ipynb
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(df['breed'].nunique(), activation='softmax')(x)
```
加上 GlobalAveragePooling 層（壓縮特徵圖），再加 Dropout 避免 overfitting，最後接上全連接層輸出所有狗品種的機率。
```ipynb
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(df['breed'].nunique(), activation='softmax')(x)
```
把整個模型接起來並編譯。
優化器：Adam 。
損失函數：分類用的 Cross-Entropy。

**開始訓練模型**
```ipynb
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)
```
開始訓練模型，最多訓練 10 次，如果驗證集三次都沒進步，就停下來並還原最佳的權重。

**畫出訓練過程圖**
```ipynb
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
```
畫出訓練跟驗證集的 Loss 變化。
畫出準確率變化。
 
 **驗證集預測 + 評估**
 ```ipynb
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
Y_pred = model.predict(valid_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = valid_generator.classes
```
對驗證集做預測，拿出每張圖預測結果跟真實標籤來做評估。
 ```ipynb
class_names = list(valid_generator.class_indices.keys())
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
```
印出分類報告（precision, recall, F1-score）。
 ```ipynb
top_20_classes = df['breed'].value_counts().head(20).index.tolist()
indices = [valid_generator.class_indices[breed] for breed in top_20_classes]
cm = confusion_matrix(y_true, y_pred)
cm_20 = cm[np.ix_(indices, indices)]
```
只看最常見的前 20 個品種，抽出對應的 confusion matrix。
 ```ipynb
plt.figure(figsize=(12, 10))
sns.heatmap(cm_20, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_20_classes,
            yticklabels=top_20_classes)
```
畫出混淆矩陣，看哪些品種最常被搞混。

**測試集預測 + 輸出結果 CSV**
 ```ipynb
test_filenames = os.listdir(test_dir)
test_df = pd.DataFrame({'filename': test_filenames})
```
把測試集的檔案名抓出來，放進一個 DataFrame。
 ```ipynb
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_dir,
    x_col='filename',
    y_col=None,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

preds = model.predict(test_generator)
```
跟訓練集一樣設置圖片生成器，但不用標籤，因為是要做預測。
對測試資料做預測，輸出每一張圖屬於各品種的機率。
 ```ipynb
breed_labels = train_generator.class_indices
breed_labels = dict((v, k) for k, v in breed_labels.items())
preds_df = pd.DataFrame(preds, columns=[breed_labels[i] for i in range(len(breed_labels))])
preds_df.insert(0, 'id', [fname[:-4] for fname in test_filenames])
```
把預測結果整理成提交格式，第一欄是圖片 ID，接著是每個品種的機率。
 ```ipynb
submission_path = 'submission.csv'
preds_df.to_csv(submission_path, index=False)
files.download(submission_path)
```
 輸出成 CSV 並下載下來，這份就是要上傳到 Kaggle 的提交檔。