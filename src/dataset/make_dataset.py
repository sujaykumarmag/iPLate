#######################################################################################################
# Author : SujayKumar Reddy M
# Project : iPLate 
# Description : Dataset Creation for models (Specially done for Extracted data)
# Sharing : Hemanth Karnati, Melvin Paulsam
# School : Vellore Institute of Technology, Vellore
# Project Manager : Prof. Dr. Swarnalatha P
#######################################################################################################

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
from torchvision.datasets import ImageFolder

class Dataset():

    def __init__(self, args):
        self.dataset = args.dataset
        self.batchsize = args.batchsize
        # Needs to be changed when the default dataset is downloaded
        if self.dataset == "default":
            self.IMAGE_DIR = "extracted_data/" 
        elif self.dataset == "augmented":
            self.IMAGE_DIR = "augmented_data/"
        else:
            self.IMAGE_DIR = "extracted_data/"
        self.classnames = os.listdir(self.IMAGE_DIR)
        self.classnames = self.remove_file_artifacts(self.classnames)


    def remove_file_artifacts(self,arr):
        '''
            Input : File array to remove the file
            Description : Only for Mac's 
        '''
        if ".DS_Store" in arr:
            arr.remove(".DS_Store")
        return arr
    


    def check_y(self,y_train,y_test):
        for i in y_test:
            if i not in y_train:
                print(i)

    

    def get_dataloaders(self):
       
        transform=transforms.Compose([
                        transforms.Resize(200),
                        transforms.ToTensor(),
                                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = ImageFolder(root=self.IMAGE_DIR+"training/",transform=transform)
        test_dataset = ImageFolder(root=self.IMAGE_DIR+"testing/",transform=transform)
        

        train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=self.batchsize)
        test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True,batch_size=self.batchsize)

        return train_loader, test_loader
    

    



