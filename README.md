

# Datasets
+ BPIC11 
+ BPIC15_1 
+ credit-card-new
+ pub-new

( Cleaned data from https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR )
+ Structure of the repository
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
   ┣ config.py
   ┗ Read.me
```

# Environment 

+ CPU : Intel i9-12900
+ GPU : Nvidia RTX-3090TI
  
This project is based on the 2.2.0-cuda11.8-cudnn8-runtime Docker image, which includes CUDA 11.8 and cuDNN 8 (Python 3.10.13).

You can download related docker image with :
```
docker pull pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
```
Libraries that are required for the experiment are in the ```requirements.txt```
```
pip install -r requirements.txt
```
# Launching Experiment


Example command : 
```
python3 main.py --dataset=BPIC11_f1 --task=next_activity
```

only next_activity, outcome task will work
