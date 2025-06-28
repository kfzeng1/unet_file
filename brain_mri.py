import os
import glob
import random
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from net2 import *
# %matplotlib inline

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import make_grid
import torchvision.transforms as tt
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
model = UNet(base_channel=32).to(device)


def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed()


ROOT_PATH = 'kaggle/input/lgg-mri-segmentation/kaggle_3m/'

mask_files = glob.glob(ROOT_PATH + '*/*_mask*')
image_files = [file.replace('_mask', '') for file in mask_files]

def diagnosis(mask_path):
    return 1 if np.max(cv2.imread(mask_path)) > 0 else 0

files_df = pd.DataFrame({"image_path": image_files,
                  "mask_path": mask_files,
                  "diagnosis": [diagnosis(x) for x in mask_files]})

# print(files_df)

# ax = files_df['diagnosis'].value_counts().plot(kind='bar', stacked=True, figsize=(6,6), color=['green', 'red'])
# ax.set_title('Data Distribution', fontsize=15)
# ax.set_ylabel('No. of Images', fontsize=15)
# ax.set_xticklabels(['No Tumor', 'Tumor'], fontsize=12, rotation=0)
# for i, rows in enumerate(files_df['diagnosis'].value_counts().values):
#     ax.annotate(int(rows), xy=(i, rows+12), ha='center', fontweight='bold', fontsize=12)
# plt.show()

train_df, val_df = train_test_split(files_df, stratify=files_df['diagnosis'], test_size=0.1, random_state=0)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df, test_df = train_test_split(train_df, stratify=train_df['diagnosis'], test_size=0.15, random_state=0)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# print("Train: {}\nVal: {}\nTest: {}".format(train_df.shape, val_df.shape, test_df.shape))


set_seed()

images, masks = [], []
df_positive = train_df[train_df['diagnosis']==1].sample(5).values

for sample in df_positive:
    img = cv2.imread(sample[0])
    mask = cv2.imread(sample[1])
    images.append(img)
    masks.append(mask)
images = np.hstack(np.array(images))
masks = np.hstack(np.array(masks))

# fig = plt.figure(figsize=(15,10))
# grid = ImageGrid(fig, 111, nrows_ncols=(3,1), axes_pad=0.4)
#
# grid[0].imshow(images)
# grid[0].set_title('Images', fontsize=15)
# grid[0].axis('off')
# grid[1].imshow(masks)
# grid[1].set_title('Masks', fontsize=15)
# grid[1].axis('off')
# grid[2].imshow(images)
# grid[2].imshow(masks, alpha=0.4)
# grid[2].set_title('Brain MRI with mask', fontsize=15)
# grid[2].axis('off')
# plt.show()

class BrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image) / 255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask) / 255.

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.float32)
        image = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))
        mask = torch.from_numpy(mask).type(torch.float32)

        return image, mask


train_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.96, 1.04), translate_percent=(-0.01, 0.01), rotate=0, p=0.25)
])

val_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
    A.HorizontalFlip(p=0.5),
])

test_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0)
])



set_seed()

train_ds = BrainDataset(train_df, train_transform)
val_ds = BrainDataset(val_df, val_transform)
test_ds = BrainDataset(test_df, test_transform)


def dataset_info(dataset):
    print(f'Size of dataset: {len(dataset)}')
    index = random.randint(1, 40)
    img, label = dataset[index]
    # print(f'Sample-{index} Image size: {img.shape}, Mask: {label.shape}\n')

# print('Train dataset:')
# dataset_info(train_ds)
# print('Validation dataset:')
# dataset_info(val_ds)
# print('Test dataset:')
# dataset_info(test_ds)


batch_size = 16

set_seed()
train_dl = DataLoader(train_ds,
                      batch_size,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True)

set_seed()
val_dl = DataLoader(val_ds,
                    batch_size,
                    num_workers=2,
                    pin_memory=True)

test_dl = DataLoader(val_ds,
                    batch_size,
                    num_workers=2,
                    pin_memory=True)


def denormalize(images):
    means = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return images * stds + means


def show_batch(dl):
    for images, masks in dl:
        fig1, ax1 = plt.subplots(figsize=(24, 24))
        ax1.set_xticks([])
        ax1.set_yticks([])
        denorm_images = denormalize(images)
        ax1.imshow(make_grid(denorm_images[:13], nrow=13).permute(1, 2, 0).clamp(0, 1))

        fig2, ax2 = plt.subplots(figsize=(24, 24))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(make_grid(masks[:13], nrow=13).permute(1, 2, 0).clamp(0, 1))
        # plt.show()
        break

def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union

def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
num_epochs = 30

def train_loop(model, loader, loss_func):
    model.train()
    train_losses = []
    train_dices = []

    #     for i, (image, mask) in enumerate(tqdm(loader)):
    for i, (image, mask) in enumerate(loader):
        image = image.to(device)
        mask = mask.to(device)
        outputs = model(image)
        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
        loss = loss_func(outputs, mask)
        train_losses.append(loss.item())
        train_dices.append(dice)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_dices, train_losses


def eval_loop(model, loader, loss_func, training=True):
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for step, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)

            outputs = model(image)
            loss = loss_func(outputs, mask)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())

            val_loss += loss
            val_dice += dice

        val_mean_dice = val_dice / step
        val_mean_loss = val_loss / step

        if training:
            scheduler.step(val_mean_dice)

    return val_mean_dice, val_mean_loss


def train_model(train_loader, val_loader, loss_func, optimizer, scheduler, num_epochs):
    train_loss_history = []
    train_dice_history = []
    val_loss_history = []
    val_dice_history = []

    for epoch in range(num_epochs):
        train_dices, train_losses = train_loop(model, train_loader, loss_func)
        train_mean_dice = np.array(train_dices).mean()
        train_mean_loss = np.array(train_losses).mean()
        val_mean_dice, val_mean_loss = eval_loop(model, val_loader, loss_func)

        train_loss_history.append(np.array(train_losses).mean())
        train_dice_history.append(np.array(train_dices).mean())
        val_loss_history.append(val_mean_loss)
        val_dice_history.append(val_mean_dice)

        print('Epoch: {}/{} |  Train Loss: {:.3f}, Val Loss: {:.3f}, Train DICE: {:.3f}, Val DICE: {:.3f}'.format(
            epoch + 1, num_epochs,
            train_mean_loss,
            val_mean_loss,
            train_mean_dice,
            val_mean_dice))

    return train_loss_history, train_dice_history, val_loss_history, val_dice_history




if __name__ == '__main__':
    train_loss_history, train_dice_history, val_loss_history, val_dice_history = train_model(train_dl, val_dl,
                                                                                             bce_dice_loss, optimizer,
                                                                                             scheduler, num_epochs)


    def plot_dice_history(model_name, train_dice_history, val_dice_history, num_epochs):
        x = np.arange(num_epochs)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x, train_dice_history, label='Train DICE', lw=3, c="b")
        plt.plot(x, val_dice_history, label='Validation DICE', lw=3, c="r")

        plt.title(f"{model_name}", fontsize=20)
        plt.legend(fontsize=12)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("DICE", fontsize=15)

        plt.show()


    plot_dice_history('UNET', train_dice_history, val_dice_history, num_epochs)




    test_dice, test_loss = eval_loop(model, test_dl, bce_dice_loss, training=False)
    print("Mean IoU/DICE: {:.3f}%, Loss: {:.3f}".format((100 * test_dice), test_loss))

    test_sample = test_df[test_df["diagnosis"] == 1].sample(24).values[0]
    image = cv2.resize(cv2.imread(test_sample[0]), (128, 128))
    mask = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

    # pred
    pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0, 3, 1, 2)
    pred = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(pred)
    pred = model(pred.to(device))
    pred = pred.detach().cpu().numpy()[0, 0, :, :]

    pred_t = np.copy(pred)
    pred_t[np.nonzero(pred_t < 0.3)] = 0.0
    pred_t[np.nonzero(pred_t >= 0.3)] = 255.
    pred_t = pred_t.astype("uint8")

    # plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("image")
    ax[0, 1].imshow(mask)
    ax[0, 1].set_title("mask")
    ax[1, 0].imshow(pred)
    ax[1, 0].set_title("prediction")
    ax[1, 1].imshow(pred_t)
    ax[1, 1].set_title("prediction with threshold")
    plt.show()

    torch.save(model.state_dict(), 'brain-mri-unet.pth1')