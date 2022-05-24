# optml-project
In this project, we question the assumptions for data heterogeneity in FedAvg analysis.

## Installation details

- We are using python 3.8.10
- The `requirements.txt` file has all modules with correct versions
- You can set it up as follows:

```
# create the virtual environment
python3 -m venv venv

# activate it
source venv/bin/activate

pip install -r requirements.txt
```

## Datasets

We will use the [leaf FL benchmark](https://github.com/TalwalkarLab/leaf) for performing tests. It contains variety of datasets aimed at different learning tasks. The current code supports the FEMNIST dataset.

LEAF has been added a submodule. To generate the dataset, do the following:

```
# pull git submodule locally
git submodule update --init --recursive

cd leaf/data/feminst

# command description at https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 100 -t sample --smplseed 10 --spltseed 10
```

This make take an hour depending on your machine. So only do this where you intend to run the code.