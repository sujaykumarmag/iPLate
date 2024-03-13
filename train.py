#######################################################################################################
# Author : SujayKumar Reddy M
# Project : iPLate 
# Description : Entry Point of the Program for Training Pipelines
# Sharing : Hemanth Karnati, Melvin Paulsam
# School : Vellore Institute of Technology, Vellore
# Project Manager : Prof. Dr. Swarnalatha P
#######################################################################################################

import pandas 
import numpy 
import torch 
import torch.nn as nn
import argparse
from src.dataset.make_dataset import Dataset
from training.normal_train import NormalTrain

parser = argparse.ArgumentParser(description="Argument Parser for the iPlate AI training model")
parser.add_argument('dataset',metavar="dataset",type=str,default="indian",
                help="The dataset needed for training 3 types (indian by VITB), (indian augmented), (total augmented)")
parser.add_argument('--validation',metavar="validation",type=str,default="combined",
                    help="The dataset needs to be evaluated using (combined dataset - default), (KFold - stratify), (indian foods provided by VITB) ")

# Training Params
parser.add_argument('--batchsize',metavar='batchsize',type=int,default=32,
                    help="Batchwise Training Hyperparameter")
parser.add_argument("--learningrate",metavar='learningrate',type=float,default=0.01,
                    help='Learning Rate Hyperparameter')
parser.add_argument('--epochs',metavar='epochs',type=int,default=100,help="Number of Epochs")
parser.add_argument("--device",metavar='device',type=str,default='cpu',help="Training device (cuda/mps)")
parser.add_argument("--numworkers",metavar="numworkers",type=int,default=0,
                    help="Number of Workers used in GPU training for the dataloader for CUDA")


args = parser.parse_args()

dataset = Dataset(args)
trainloader, testloader = dataset.get_dataloaders()
NormalTrain(args,trainloader,testloader).train()
for u in trainloader:
    sample, label = u

print(sample.shape)
print(label)





