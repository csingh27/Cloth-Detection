# IMPORTING LIBRARIES


# * General libraries
import cv2
import os
import glob
import pandas as pd
import numpy as np
import shutil
from shutil import copyfile

# * ML specific libraries
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn import preprocessing, model_selection

# Function to split training and test set

def dataset_split(df, folder, train_img_path, train_label_path):
	filenames = []
	for name in df.file_name:
		filenames.append(name)
	# Directory Structure :
	# --Dataset_yolo
	# 	 --Images
	# 	 	--Train
	# 	 	--Val
	# --Dataset_yolo
	# 	 --Labels
	# 	 	--Train
	# 	 	--Val
	# Image format .jpg, Label format .txt (Separate .txt file label for each image)
	# Inside label.txt : x_center_norm, y_center_norm, width_norm, height_norm
	for filename in filenames:
		yolo_list = []
		for i,row in df[df.file_name == filename].iterrows():
			yolo_list.append([0,row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])
		yolo_list = np.array(yolo_list)
		print("\n",yolo_list)
		txt_filename = os.path.join(train_label_path,str(row.file_name.split('.')[0])+".txt")
		print("\n",txt_filename)
		np.savetxt(txt_filename, yolo_list, fmt=["%d","%f","%f","%f","%f"])
		shutil.copyfile(os.path.join(folder,row.file_name), os.path.join(train_img_path,row.file_name))

	return(0)

# Function to convert Dataset into YoloV5 compatible format 

def convert_dataset(path,table):
	# YoloV5 compatible dataset has x_center_norm, y_center_norm, width_norm, height_norm as its columns
	img_width = 224
	img_height = 224
	width=[]
	height=[]
	x_center=[]
	y_center=[]
	df = pd.DataFrame(columns=['file_name','x_center_norm','y_center_norm','width_norm','height_norm'])
	df["file_name"]=table['Unnamed: 0'].astype(str)
	df["width_norm"]=(table["2"]-table["0"])/img_width
	df["height_norm"]=(table["3"]-table["1"])/img_height
	df["x_center_norm"]=(table["0"]/img_width)+(df["width_norm"]/2)
	df["y_center_norm"]=(table["1"]/img_height)+(df["height_norm"]/2)
	print(df)
	df.to_csv(os.path.join(path,'Dataset/Dataset_yolo/BB_labels_yolo.txt'))
	return(df)

# Function to load dataset

def display_dataset_images(folder, table):
	print(table)
	images=[]
	image_path=[]
	filename=[]
	x1=[]
	y1=[]
	x2=[]
	y2=[]
	start_point=[]
	end_point=[]
	i=0
	print("Displaying dataset images ... \n")
	for i in range(len(table.index)):
		print("Image",i)
		image_path.append(table.iloc[i,0])
		image_path[i]=os.path.join(folder,image_path[i])
		print(image_path[i])
		# Gets xmin, ymin, xmax, ymax when Table (dataset_raw) is passed as argument
		# Gets x_center_norm, y_center_norm, width_norm, height_norm when DF (dataset_yolo) is passed as argument
		x1.append(table.iloc[i,1])
		y1.append(table.iloc[i,2])
		x2.append(table.iloc[i,3])
		y2.append(table.iloc[i,4])
		# De-normalizes in the case when DF is passed as argument
		if "x_center_norm" in table:
			image_path[i]=os.path.join(folder,table.iloc[i,0])
			print(image_path[i])
			x1[i]=int(224*(x1[i]-x2[i]/2)) #224(x_center-width/2)
			y1[i]=int(224*(y1[i]-y2[i]/2)) #224(x_center-height/2)
			x2[i]=int(224*x2[i]+x1[i]-x2[i]/2) #224(width+x_center-width/2)
			y2[i]=int(224*y2[i]+y1[i]-y2[i]/2) #224(width+x_center-height/2)
		start_point.append((x1[i],y1[i]))
		print("Bounding box \n(xmin,ymin)",start_point[i])
		end_point.append((x2[i],y2[i]))
		print("(xmax,ymax)",end_point[i])
		img=cv2.imread(image_path[i])
		img = cv2.rectangle(img, start_point[i],end_point[i], color=(255, 0, 0), thickness=2) 
		cv2.waitKey(100)
		# Displays images
		cv2.imshow('image',img)
		if img is not None:
			images.append(img)
		i=i+1
	cv2.waitKey(500)
	return(images,start_point,end_point)



# Define main function

def main():
    # Define path of the dataset
    path=os.path.dirname(os.path.abspath(__file__))
    print("Current directory : ")
    print(path)
    folder=os.path.join(path,"Dataset/Dataset_raw")
    os.chdir(path)
    # Load dataset
    csv_path=os.path.join(folder,"BB_labels.csv")
    print(path)
    table = pd.read_csv(csv_path) 
    print(table)
    # Display dataset
    print("Raw dataset ... \n")
    display_dataset_images(folder,table)
    # Convert dataset to Yolo Compatible
    df=convert_dataset(path,table)
    # Train-test split
    df_train, df_valid = model_selection.train_test_split(df, test_size=0.2, random_state=13, shuffle=True)
    train_img_path = os.path.join(path,'Dataset/Dataset_yolo/images/train')
    train_label_path = os.path.join(path,'Dataset/Dataset_yolo/labels/train')
    valid_img_path = os.path.join(path,"Dataset/Dataset_yolo/images/val")
    valid_label_path = os.path.join(path,"Dataset/Dataset_yolo/labels/val")
    dataset_split(df_train, folder, train_img_path, train_label_path)
    dataset_split(df_valid, folder, valid_img_path, valid_label_path)
    print("No. of Training images", len(os.listdir(train_img_path)))
    print("No. of Training labels", len(os.listdir(train_label_path)))
    print("No. of valid images", len(os.listdir(valid_img_path)))
    print("No. of valid labels", len(os.listdir(valid_label_path)))
    #Display converted dataset for verification
    print("Train dataset ... \n")
    display_dataset_images(train_img_path,df_train)
    print("Validation dataset ... \n")
    display_dataset_images(valid_img_path,df_valid)

if __name__ == '__main__':
    main()

