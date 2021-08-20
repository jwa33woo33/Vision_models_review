import os
import math
from datetime import datetime

import numpy as np

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from voc import VOCDataset
from darknet import DarkNet
from yolo_v1 import YOLOv1
from loss import Loss

import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if GPU device are available
use_gpu = torch.cuda.is_available()
assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA.'
print(f'CUDA current_device: {torch.cuda.current_device()}')
print(f'CUDA device_count: {torch.cuda.device_count()}')

# Path to data DIR.
image_dir = 'data/VOC2012/JPEGImages/'

# Path to label files.
train_label = ('data/voc2012.txt') # ('data/voc2007.txt', 'data/voc2012.txt')
# val_label = 'data/voc2007test.txt'

# Path to checkpoint file containing pre-trained DarkNet weight.
# checkpoint_path = 'weights/darknet/model_best.pth.tar'

# Frequency to print/log the results
print_freq = 5
tb_log_freq = 5

#  Training Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 32

# Learning rate scheduling
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#  Load pre-trained darknet.
darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
# darknet.features = torch.nn.DataParallel(darknet.features)

# src_state_dict = torch.load(checkpoint_path)['state_dict']
# dst_state_dict = darknet.state_dict()

# for k in dst_state_dict.keys():
#     print('Loading weight of', k)
#     dst_state_dict[k] = src_state_dict[k]
# darknet.load_state_dict(dst_state_dict)

# Load YOLO model.
yolo = YOLOv1(darknet.features)
# yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layers)
yolo.to(device)

# Setup loss and optimizer.
criterion = Loss(feature_size=yolo.feature_size)
optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

#  Load Pascal-VOC dataset
train_dataset = VOCDataset(True, image_dir, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# val_dataset = VOCDataset(False, image_dir, val_label)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Number of training images:', len(train_dataset))

# Open TensorBoardX summary writer
log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('result/yolo', log_dir)
writer = SummaryWriter(log_dir=log_dir)

#  Training loop
logfile = open(os.path.join(log_dir, 'log.txt'), 'w')
best_val_loss = np.inf

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch} / {num_epochs}')

    #  Training.
    yolo.train()
    total_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(train_loader):
        # Update learning rate.
        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        lr = get_lr(optimizer)

        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = imgs.to(device)
        targets = targets.to(device)

        #  Forward to compute loss.
        preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        #  Backward to update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss.
        if i % print_freq == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Iter [{i}/{len(train_loader)}], LR:{lr:.4f}, Loss:{loss_this_iter:.4f}, Average Loss:{total_loss / float(total_batch)}')
        
        # TensorBoard.
        n_iter = epoch * len(train_loader) + i
        if n_iter % tb_log_freq == 0:
            writer.add_scalar('train/loss', loss_this_iter, n_iter)
            writer.add_scalar('lr', lr, n_iter)

        # Validation.
        yolo.eval()
        val_loss = 0.0
        total_batch = 0

        # for i, (imgs, targets) in enumerate(val_loader):
        #     # Load data as a batch.
        #     batch_size_this_iter = imgs.size(0)
        #     imgs = imgs.to(device)
        #     targets = targets.to(device)

        #     # Forward to compute validation loss
        #     with torch.no_grad():
        #         preds = yolo(imgs)
        #     loss = criterion(preds, targets)
        #     loss_this_iter = loss.item()
        #     val_loss += loss_this_iter * batch_size_this_iter
        #     total_batch += batch_size_this_iter
        # val_loss /= float(total_batch)

        # Save results
        # logfile.writelines(str(epoch + 1) + '\t' + str(val_loss) + '\n')
        # logfile.flush()
        # torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_latest.pth'))
        # if best_val_loss > val_loss:
        #     best_val_loss = val_loss
        #     torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))


        # # Print
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Best val Loss: {best_val_loss:.4f}')

        # # TensorBoard.
        # writer.add_scalar('test/loss', val_loss, epoch + 1)

writer.close()
logfile.close()