# ContrastivePose
Codes for experiments on CalMS21 dataset is provided in CalMS21/ directory, and experiments on in-house dataset in in_house_data/

Set up python environment with conda
```
conda env create -f ContrastivePose.yml
```

To run experiments, first modify the following three variables run.py script
```
train = True
test = "finetune" # choice: contrastive, finetune, from_scratch
experiment = 1
```
* to train network, set train = True, or else the mode will be inference only
* test mode can be one of the following: contrastive, finetune or from_scratch. Select finetune for best results. For more details, refer to methods section of paper
* experiment number is to save or load the model version. Change number to save a new model version

Then run the following in command line
```
python run.py
```

For CalMS21, please download the data first from https://data.caltech.edu/records/s0vdx-0k302, task1_classic_classification.zip. Copy the csv files into the datasets directory under trainset and testset
```
CalMS21/datasets/
           |- trainset/
           |- testset /
```