# Importing libraries

import cv2
import os
import glob

# Function to load dataset

def load_dataset_images(folder):
	images=[]
	img_path=os.listdir(folder)
	print(img_path)
	for filename in glob.glob(os.path.join(folder,'*.jpg')):
		print(filename)
		img=cv2.imread(filename)
		cv2.waitKey(50)
		cv2.imshow('image',img)
		if img is not None:
			images.append(img)
	return images

# Define path of the dataset

path=os.path.dirname(os.path.abspath(__file__))
print("Current directory : ")
print(path)
folder=os.path.join(path,"Dataset_BB")

# Load dataset
load_dataset_images(folder)

