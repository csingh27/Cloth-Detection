# IMPORTING LIBRARIES


# * General
import cv2
import os
import glob
import pandas as pd


# * ML framework
import torch
import torchvision
from torch.utils.data import DataLoader


# * ML model


# Function to convert Dataset into YoloV5 compatible format 

def convert_dataset(folder,table):
	img_width = 224
	img_height = 224
	width=[]
	height=[]
	x_center=[]
	y_center=[]
	df = pd.DataFrame(columns=['x_center_norm','width_norm','y_center_norm','height_norm'])
	df["width_norm"]=(table["2"]-table["0"])/img_width
	df["height_norm"]=(table["3"]-table["1"])/img_height
	df["x_center_norm"]=(table["0"]+df["width_norm"])/img_width
	df["y_center_norm"]=(table["1"]+df["height_norm"])/img_height
	print(df)
	df.to_csv(os.path.join(folder,'BB_labels_yolo.txt'))
	return()

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
	os.chdir(folder)
	print("Displaying dataset images ... \n")
	for i in range(len(table.index)):
		print("Image",i)
		image_path.append(table.loc[i,"Unnamed: 0"])
		image_path[i]=os.path.join(os.getcwd(),image_path[i])
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
    folder=os.path.join(path,"Dataset_BB")
    # Load dataset
    csv_path=os.path.join(folder,"BB_labels.csv")
    table = pd.read_csv(csv_path) 
    print(table)
    # Display dataset
    display_dataset_images(folder,table)
    # Convert dataset to Yolo Compatible
    convert_dataset(folder,table)



if __name__ == '__main__':
    main()


#models=torchvision.models.detectron.faster_rcnn_resnet150_fpm(pretrained=True)
#train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])


