"""
Code for generating pseudolabels
"""

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import os
import argparse
import numpy as np

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from core.data.ce_dataset import CEDataset

from utils_semisup import get_model

parser = argparse.ArgumentParser(
    description='Apply standard trained model to evaluate its performance on unconditional CEs')
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='name of the model')
parser.add_argument('--model_path', type=str,
                    help='path of checkpoint to standard trained model')

parser.add_argument('--class_num', type=int, default=10)

parser.add_argument('--data_dir', default='./ce_data/', type=str,
                    help='directory that has CE data')

parser.add_argument('--dataset_name', default='out_cifar10_0_49999_ce5', type=str, help='name of the CE dataset in data_dir')

parser.add_argument('--batch_size', default=5000, type=int,
                    help='batch size of testing')



parser.add_argument('--base_dataset', default='~/data/cifar10', type=str, help='base dataset path. used for comparison')
args = parser.parse_args()


if not os.path.exists(args.model_path):
    raise ValueError('Model %s not found' % args.model_path)

# Loading model
checkpoint = torch.load(args.model_path)
num_classes = checkpoint.get('num_classes', args.class_num)
normalize_input = checkpoint.get('normalize_input', False)
model = get_model(args.model, 
                  num_classes=num_classes,
                  normalize_input=normalize_input)
model = nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

Fsoftmax = torch.nn.functional.softmax


# load in the CEDataset

dataset = CEDataset(args.data_dir, args.dataset_name, transform=ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)


num_correct = 0

for i, batch in enumerate(dataloader):

    print(i+1, " of ", len(dataloader))

    x, y = batch
    x = x.cuda()
    y = y.cuda()
    with torch.no_grad():
        cons = Fsoftmax(model(x), dim=1)
    batch_correct = (torch.max(cons, dim=1)[1] == y).sum()
    num_correct += batch_correct


print("Accuracy {:1.3f}%".format(
    num_correct / len(dataset) * 100.
))

