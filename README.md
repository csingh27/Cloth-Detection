# Coding-Challenge-Sewts  

*Run :*   
python3 train.py

*Goal :*   
Object (towel) localization (Bounding Box Detection)  

*Given :*  
Dataset of 
* 100 images with towel  
* Bounding box labels for each image in CSV  

*Steps :*  
1. Load dataset images (JPEG) and BB_labels (TXT) files✓  
2. Analyse and display dataset✓  
3. Model selection (Yolo v5)✓  
4. Convert dataset to a model compatible format✓  
5. Train-test split✓  
6. Data pre-processing✓  
7. Model training✓  
9. Inference✓  
 
*Package Installation requirements :*  
Torch -  
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
Opencv -  
pip3 install opencv-python  
Yolo v5 -  
git clone  'https://github.com/ultralytics/yolov5.git'  
Pandas -  
pip3 install pandas
YAML -  
pip3 install pyyaml
Tensorboard -  
pip3 install tensorboard
Matplotlib -  
pip3 install matplotlib

*Model Training :*   
* Train model
python3 train.py --img 224 --batch 4 --epochs 2     --data /home/chandandeep/GitHub/Coding-Challenge-Sewts/Model/bcc.yaml --cfg /home/chandandeep/GitHub/Coding-Challenge-Sewts/Model/yolov5s.yaml --name BCCM
* Visualize using Tensorboard
tensorboard --logdir=runs

*Model Inference :*   
python yolov5/detect.py --source ~/Coding-Challenge-Sewts/Dataset/Dataset_yolo/images/val  --weights '~/yolov5/runs/train/BCCM18/weights/best.pt'


*Challenge Evaluation criteria :*   

• Sophisticated naming conventions  
• Code styling (i.e. PEP8)  
• Simple but expressive comments  
• Code reusability  
• Testing procedures  
• Consistent use of a version control system (i.e. git)  




