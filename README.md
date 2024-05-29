# Datasets
+ BPIC11 
+ BPIC15_1 
+ credit-card-new ( Synthetic )
+ pub-new ( Synthetic )

You can have BPIC15_1, Credit, Pub at below link.
Please locate them in data/dataset/ folder
https://drive.google.com/file/d/1KQL7WCyH_oIDUkt8V4zDFEl_KXD-o27N/view?usp=sharing

# Structure of the Repository
``` 
Imperfection_Pattern
 ┣ requirements.txt
 ┣ main.py
 ┣ data
 ┃ ┣ preprocess.py
 ┃ ┗ datasets
 ┃   ┣ BPIC11
 ┃   ┣ BPIC15_1 (will  be uploaded)
 ┃   ┣ pub-new (will  be uploaded)
 ┃   ┗ credit-card-new (will  be uploaded)
 ┣ models
 ┃ ┗ model.py
 ┗ utils
   ┗ config.py
```

# Environment 

+ CPU : Intel i9-12900
+ GPU : Nvidia RTX-3090TI
  
This project is based on the 2.2.0-cuda11.8-cudnn8-runtime Docker image, which includes CUDA 11.8 and cuDNN 8 (Python 3.10.13).

You can download related docker image with :
```
docker pull pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
```
!! If your graphics card's driver version doesn't match the CUDA version or if the CUDA version doesn't match the PyTorch version, it may be not able to train the model using the GPU. 
(However, it is set up to train on the CPU instead in such cases.)

Libraries that are required for the experiment are in the ```requirements.txt```
```
pip install -r requirements.txt
```
# Launching Experiment


Example command : 
```
python3 main.py --dataset_name=BPIC11_f1 --task=next_activity
```

only next_activity, outcome task will work
