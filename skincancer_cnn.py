from numpy.random import seed
seed(1)
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import itertools

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

"""
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import ResNet152, Xception,VGG16,EfficientNetB4
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,AveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
"""

np.random.seed(123)
print("done")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


import glob, os
lesion_type_dict = {'akiec': 'Actinic keratoses',
                    'bcc': 'Basal cell carcinoma',
                    'bkl': 'Benign keratosis-like lesions ',
                    'df': 'Dermatofibroma',
                    'nv': 'Melanocytic nevi',
                    'mel': 'Melanoma',
                    'vasc': 'Vascular lesions'}
imageid_path_dict = {}

os.chdir("S:\DataScience\skin-cancer-mnist-ham10000\HAM10000_images_part_1")
for x in glob.glob("*.jpg"):
    # print(x)
    imageid_path_dict[
        x[:-4]] = "S:\DataScience\skin-cancer-mnist-ham10000\HAM10000_images_part_1/" + x

os.chdir("S:\DataScience\skin-cancer-mnist-ham10000\HAM10000_images_part_2")
for x in glob.glob("*.jpg"):
    # print(x)
    imageid_path_dict[
        x[:-4]] = "S:\DataScience\skin-cancer-mnist-ham10000\HAM10000_images_part_2/" + x


print(imageid_path_dict)
# print(glob(os.path.join(base_skin_dir,'*.jpg')))
skin_df = pd.read_csv('S:\DataScience\skin-cancer-mnist-ham10000\HAM10000_metadata.csv')
print(skin_df)
skin1 = []
skin2 = []
for x in skin_df['image_id']:
    if x in imageid_path_dict:
        skin1.append(imageid_path_dict.get(x))

skin_1 = pd.DataFrame(skin1)
skin_df['path'] = skin_1

for x in skin_df['dx']:
    if x in lesion_type_dict:
        skin2.append(lesion_type_dict.get(x))

skin_2 = pd.DataFrame(skin2)
skin_df['cell_type'] = skin_2

skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
skin_df.groupby(['dx']).count()

print("*************************************************************************")
print(skin_df['path'])
print("*************************************************************************")
print(skin_df['cell_type'])
print("*************************************************************************")
print(skin_df['cell_type_idx'])

#####

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
ax=sns.countplot(x="dx", data=skin_df,palette = "cool")
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
#plt.show()
ax=sns.countplot(x="sex", data=skin_df,palette = "hot")
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
####

print('using median filter and unsharp masking the data...')

from skimage.filters import unsharp_mask
skin_df['image'] = skin_df['path'].map(lambda x: unsharp_mask(cv2.medianBlur(np.asarray((Image.open(x).resize((120,120)))),5)))

print("Extracting features and target.....")
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']
print('done')

fig = plt.figure(figsize=(10, 10))
for i in range(1, 10):
    img = np.random.randint(0, 10014)
    fig.add_subplot(3, 3, i)
    plt.imshow(skin_df['image'][img])
    plt.title(skin_df['cell_type'][img], fontdict={'fontsize': 18})
    plt.axis('off')

#plt.show()

#from tensorflow.keras.utils import to_categorical
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.2,random_state=666)
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)
x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 999)
print(x_test_o)

x_train = x_train.reshape(x_train.shape[0], *(120, 120, 3))
x_test = x_test.reshape(x_test.shape[0], *(120, 120, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(120, 120, 3))


#hier die funktion für zsm
class Matcher(Dataset):

    def __init__(self, x,y):


        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x, self.y


input_shape = (120, 120,3)
num_classes = 7

#hier dei drei splits
train_data = Matcher(x_train, y_train)
valid_data = Matcher(x_validate, y_validate)
test_data = Matcher(x_test, y_test)

#hier den data loader
batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Dropout(0,25),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Dropout(0, 25),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Dropout(0, 25),
            torch.nn.Flatten(),
            torch.nn.Linear(14400,128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, num_classes),
            torch.nn.Softmax()
        )

    def forward(self, x):
        out = self.main(x)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = CNN()
model.to(device)
print(summary(model, (3, 120, 120)))

#hier opti
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
num_epochs = 35

optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

# hier epochs

#%%time
# keeping-track-of-losses
train_losses = []
valid_losses = []

for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0

    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # validate-the-model
    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

# hier eval
# test-the-model
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for image, labels in valid_loader:
        image = image.to(device)
        labels = labels.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# Save
torch.save(model.state_dict(), 'model.h5')
