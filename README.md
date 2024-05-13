# Datasets
+ BPIC11 : will be changed into cleaned one (TBC) 
+ BPIC15_1 : will be changed into cleaned one (TBC) 
+ credit-card-new
+ pub-new

( Cleaned data from https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR )
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
# Launching Experiment
## Generating Train/ Test set

Argument Parser for generating Train/ Test set 
+ --dataset : name of your csv file ( choices : "BPIC11.csv", "BPIC15_1.csv", "credit-card-new.csv", "pub-new.csv" )
+ --task : PPM task ( choices : "next_activity", "outcome", "event_remaining_time", "case_remaining_time" )
+ (Optional) --filepath : dataset location ( default='data/datasets/' )
+ (Optional) --saveloc : processed train/ test set save location ( default='data/processed/' )

Example command : 
```
python3 dataset_gen.py --dataset=BPIC15_1.csv --task=case_remaining_time
```

<img width="241" alt="1213123" src="https://github.com/brucks1217/Imperfection-pattern/assets/112471517/b7d0718d-b8fd-42b5-b7e1-9ff6ecc3520a">
<img width="293" alt="21512512312" src="https://github.com/brucks1217/Imperfection-pattern/assets/112471517/f059841e-14d0-4e70-a2ff-d0124c960dff">
