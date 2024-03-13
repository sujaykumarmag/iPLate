#######################################################################################################
# Author : SujayKumar Reddy M
# Project : iPLate 
# Description : Dataset Split Creation for models (Specially done for Extracted data)
# Sharing : Hemanth Karnati, Melvin Paulsam
# School : Vellore Institute of Technology, Vellore
# Project Manager : Prof. Dr. Swarnalatha P
#######################################################################################################

# @Caution : Run this file separately

import os
import cv2
import numpy as np
from PIL import Image

class DatasetSplit():

    def __init__(self, image_dir):
        self.IMAGE_DIR = image_dir
        self.classnames = os.listdir(os.path.join(self.IMAGE_DIR, "images"))
        self.classnames = self.remove_file_artifacts(self.classnames)

    def remove_file_artifacts(self, arr):
        if ".DS_Store" in arr:
            arr.remove(".DS_Store")
        return arr

    def datasplit_extracted(self):
        X = []
        y = []
        test_X = []
        test_y = []

        for i, classname in enumerate(self.classnames):
            ims = os.listdir(os.path.join(self.IMAGE_DIR, "images", classname))
            ims = self.remove_file_artifacts(ims)

            for j, im in enumerate(ims):
                img_path = os.path.join(self.IMAGE_DIR, "images", classname, im)
                img = Image.open(img_path)
                img_resized = img.resize((200, 200))  # Resize the image
                img_array = np.array(img_resized)  

                if j == 2: 
                    test_X.append(img_array)
                    test_y.append(classname)
                else:
                    X.append(img_array)
                    y.append(classname)

        return X, y, test_X, test_y     

    def save_test(self):
        X, y, test_X, test_y = self.datasplit_extracted()

        # Create directories for training and testing
        train_dir = os.path.join(self.IMAGE_DIR, "training")
        test_dir = os.path.join(self.IMAGE_DIR, "testing")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save training images
        for image, label in zip(X, y):
            class_dir = os.path.join(train_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            image_path = os.path.join(class_dir, f"{label}_{len(os.listdir(class_dir))}.png")
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save testing images
        for image, label in zip(test_X, test_y):
            class_dir = os.path.join(test_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            image_path = os.path.join(class_dir, f"{label}_{len(os.listdir(class_dir))}.png")
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

IMAGE_DIR = "../../extracted_data/"
ds = DatasetSplit(IMAGE_DIR)
ds.save_test()
classnames_train = os.listdir(IMAGE_DIR+"training/")
print(len(classnames_train))

classnames_test = os.listdir(IMAGE_DIR+"testing/")
print(len(classnames_test))
