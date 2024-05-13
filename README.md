# Important Notice : 
Currently, for reading csv file the script uses "Pandas" module.

However, the memory (RAM) usage efficiency of "Pandas" is terrible.

Tried to use "Numpy" instead, but still it uses lots of memory while processing sequence encoding ( Prefx length + Zero padding ),

so terminal kills the operation. ( Only in BPIC11 )

Hence, I'm working on processing sequence encoding using "PyArrow" library or "Datatable" library. 

Moreover, current datasets in this github repo are not cleaned ( BPIC11, BPIC15_1 )
and also, don't have outcome label.

So, it will be substituted with cleaned one from  : https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR



# Datasets
+ BPIC11 : will be substituted with cleaned one (TBC) 
+ BPIC15_1 : will be substituted with cleaned one (TBC) 
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
+ + Remind that "outcome" task is only available with "BPIC15_1.csv" dataset
+ (Optional) --filepath : dataset location ( default='data/datasets/' )
+ (Optional) --saveloc : processed train/ test set save location ( default='data/processed/' )

Example command : 
```
python3 dataset_gen.py --dataset=BPIC15_1.csv --task=case_remaining_time
```

To modify which attributes to use for your experiment, please refer to utils/config.py
You have to clarify type and column name for it. ( Do not use "int" type. use "float" instead )

<img width="241" alt="1213123" src="https://github.com/brucks1217/Imperfection-pattern/assets/112471517/b7d0718d-b8fd-42b5-b7e1-9ff6ecc3520a">
<img width="293" alt="21512512312" src="https://github.com/brucks1217/Imperfection-pattern/assets/112471517/f059841e-14d0-4e70-a2ff-d0124c960dff">

Additionally, to increase the performance of PPM model, I manually added the time feature ( "time since case start", and "Time since last event").
But remind that this will be modified during substituting the event logs ( BPIC11, BPIC15 with cleaned one )

## Training/ Testing PPM model

Argument Parser for training/ testing model
+ --dataset : name of your csv file ( choices : "BPIC11.csv", "BPIC15_1.csv", "credit-card-new.csv", "pub-new.csv" )
+ --task : PPM task ( choices : "next_activity", "outcome", "event_remaining_time", "case_remaining_time" )
+ (Optional) --meta : just meta info of train/test set (output dimension,max prefix length,number of unique activities,number of attributes)
+ (Optional) --filepath : processed train/ test set location ( default='data/processed/' )
+ (Optional) --modelpath : model save path (default='models/model.pt')
+ (Optional) --batchsize : batchsize for input ( default = 32 )
+ (Optional) --lr : learning rate for training ( default = 0.001)
+ (Optional) --hiddendim : size of hidden layer for lstm model ( default = 128 )
+ (Optional) --embdim : size of embedding layer for processing "activity" ( default = 32 ) 
+ (Optional) --epoch : number of epoch for training ( default = 100 ) * Remind that there's early stopping while training

Example command : 
```
python3 main.py --dataset=BPIC15_1.csv --task=case_remaining_time
```

