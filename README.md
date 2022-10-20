# ContrastivePose
Codes for experiments on CalMS21 dataset is provided in CalMS21/ directory, and experiments on in-house dataset in in_house_data/

Set up python environment with conda
```
conda env create -f ContrastivePose.yml
```

To run experiments, run the following in command line
```
python run.py
```

For CalMS21, please download the data first from https://data.caltech.edu/records/s0vdx-0k302, task1_classic_classification.zip. Copy the csv files into the datasets directory
```
datasets/
    |- trainset/
    |- testset /
```