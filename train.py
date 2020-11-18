# IMPORTING LIBRARIES


# * General
import cv2
import os
import glob
import pandas as pd
import numpy as np
import shutil
from shutil import copyfile

# * ML framework
import torch
import torchvision
from torch.utils.data import DataLoader

# * ML model
from sklearn import preprocessing, model_selection

# Function to split training and test set


# Function to convert Dataset into YoloV5 compatible format 
def dataset_split(df, folder, train_img_path, train_label_path):
	filenames = []
	
	for name in df.file_name:
		filenames.append(name)
	for filename in filenames:
		yolo_list = []
		for _,row in df[df.file_name == filename].iterrows():
			yolo_list.append([row.file_name, row.x_center_norm, row.width_norm, row.y_center_norm, row.height_norm])
		yolo_list = np.array(yolo_list)
		print("\n",yolo_list)
		txt_filename = os.path.join(train_label_path,str(row.file_name.split('.')[0])+".txt")
		print("\n",txt_filename)
		np.savetxt(txt_filename, yolo_list, fmt="%s")
		shutil.copyfile(os.path.join(folder,row.file_name), os.path.join(train_img_path,row.file_name))



	return(0)


def convert_dataset(path,table):
	img_width = 224
	img_height = 224
	width=[]
	height=[]
	x_center=[]
	y_center=[]
	df = pd.DataFrame(columns=['file_name','x_center_norm','width_norm','y_center_norm','height_norm'])
	df["file_name"]=table["Unnamed: 0"]
	df["width_norm"]=(table["2"]-table["0"])/img_width
	df["height_norm"]=(table["3"]-table["1"])/img_height
	df["x_center_norm"]=(table["0"]+df["width_norm"])/img_width
	df["y_center_norm"]=(table["1"]+df["height_norm"])/img_height
	print(df)
	df.to_csv(os.path.join(path,'Dataset/Dataset_yolo/BB_labels_yolo.txt'))
	return(df)

# Function to load dataset

def display_dataset_images(folder, table):
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
		image_path.append(table.loc[i,"Unnamed: 0"])
		image_path[i]=os.path.join(folder,image_path[i])
		x1.append(table.loc[i,"0"])
		y1.append(table.loc[i,"1"])
		x2.append(table.loc[i,"2"])
		y2.append(table.loc[i,"3"])
		start_point.append((x1[i],y1[i]))
		print("Bounding box \n(xmin,ymin)",start_point[i])
		end_point.append((x2[i],y2[i]))
		print("(xmax,ymax)",end_point[i])
		img=cv2.imread(image_path[i])
		img = cv2.rectangle(img, start_point[i],end_point[i], color=(255, 0, 0), thickness=2) 
		cv2.waitKey(1)
		cv2.imshow('image',img)
		if img is not None:
			images.append(img)
		i=i+1
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
    display_dataset_images(folder,table)
    # Convert dataset to Yolo Compatible
    df=convert_dataset(path,table)
    # Training-validation split
    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
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

if __name__ == '__main__':
    main()


#models=torchvision.models.detectron.faster_rcnn_resnet150_fpm(pretrained=True)
#train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])


