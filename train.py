# Importing libraries

import cv2
import os
import glob
import pandas as pd

# Function to draw bounding boxes

def draw_bounding_box():
	csv_path=os.path.join(folder,"BB_labels.csv")
	table = pd.read_csv(csv_path) 
	x1=[]
	y1=[]
	x2=[]
	y2=[]	
	start_point=[]
	end_point=[]
	for i in range(len(table.index)):
		x1.append(table.loc[i,"0"])
		y1.append(table.loc[i,"1"])
		x2.append(table.loc[i,"2"])
		y2.append(table.loc[i,"3"])
		start_point.append((x1[i],y1[i]))
		end_point.append((x2[i],y2[i]))
	return(start_point,end_point)

# Function to load dataset

def load_dataset_images(folder):
	images=[]
	i=0
	for filename in sorted(glob.glob(os.path.join(folder,'*.jpg'))):
		print(filename)		
		img=cv2.imread(filename)
		start_point, end_point=draw_bounding_box()
		print(start_point[i])
		print(end_point[i])
		img = cv2.rectangle(img, start_point[i],end_point[i], color=(255, 0, 0), thickness=2) 
		i=i+1
		cv2.waitKey(500)
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


