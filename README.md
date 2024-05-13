+ Structure of the repository
``` 
Imperfection_Pattern
 ┣ requirements.txt
 ┣ dataset_gen.py
 ┣ main.py
 ┣ data
 ┃ ┣ loader.py
 ┃ ┣ preprocess.py
 ┃ ┗ datasets
 ┃   ┣ BPIC11
 ┃   ┣ BPIC15_1
 ┃   ┣ pub-new
 ┃   ┗ credit-card-new
 ┣ models
 ┃ ┗ model.py
 ┗ utils
   ┣ config.py
   ┗ Read.me
```

# Environment 

+ CPU : Intel i9-12900
+ GPU : Nvidia RTX-3090TI
  
This project is based on the 2.2.0-cuda11.8-cudnn8-runtime Docker image, which includes CUDA 11.8 and cuDNN 8 (Python 3.10.13).

You can download required docker image with :
```
docker pull pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
```
Libraries that are required for the experiment are in the ```requirements.txt```
```
pip install -r requirements.txt
```
