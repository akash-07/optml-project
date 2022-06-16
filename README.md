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
### (A) Synthetic
The synthetic dataset has already been generated and is included in the repository due to its small size. It is present at:

```
synthetic_data/{train, test}/{train.json, test.json}
```

The code which generates this data can be found in the `synthetic_data_generation` folder.
### (A) FEMNIST
We generate the FEMNIST dataset from the [leaf FL benchmark](https://github.com/TalwalkarLab/leaf). LEAF has been added a submodule of this repository. To perform the generation, do the following:

```bash
# pull git submodule locally
git submodule update --init --recursive

cd leaf/data/feminst

# command description at https://github.com/TalwalkarLab/leaf/tree/master/data/femnist

# To generate iid data
./preprocess.sh -s iid --sf 1.0 -k 100 -t sample --smplseed 10 --spltseed 10
# This creates the folders leaf/data/femnist/data/{train, test}
# Please rename the folders to leaf/data/femnist/data/{train_iid, test_iid} to be able to run the generation again for non-iid data

# To generate non-iid data
# Delete all folders in leaf/data/femnist/data
# Then run
./preprocess.sh -s niid --sf 1.0 -k 100 -t sample --smplseed 10 --spltseed 10
# This again creates the folders leaf/data/femnist/data/{train, test}
# Please rename the folders to leaf/data/femnist/data/{train_niid, test_niid}
```

At this point the directory stucture should be as follows:
```
+-- leaf
|  +-- data
|  |  +-- femnist
|  |  |  +-- data
|  |  |  |  +-- train_iid
|  |  |  |  +-- test_iid
|  |  |  |  +-- train
|  |  |  |  +-- test
```
The dataset generation may take an hour depending on your machine. So please only do this where you intend to run the code.

## Instructions for running FedAvg on the FEMNIST dataset

```bash
cd fedavg/src_realworld
# Activate the virtual environment as described above

# For iid data
./myrun.sh femnist 0.02 20 25 25 250 1 1 iid

# The logs will be produced in logs/fedavg/realworld/femnist/iid/25_25_lr0.02/r1 directory

# For niid data
./myrun.sh femnist 0.02 20 25 25 250 1 1 niid

# The logs will be produced in logs/fedavg/realworld/femnist/niid/25_25_lr0.02/r1 directory
```
The generated logs contain `train.csv` and `test.csv` which record the losses and accuracies of the server model after each communication round.

## Instructions for running FedAvg on the Synthetic dataset

```bash
cd fedavg/src_synthetic
# Activate the virtual environment as described above

# For 1000 local steps
./myrun.sh synthetic 0.2 100 1000 1000 50 1 1

# Change accordingly for different number of local steps {10, 50, 300}
./myrun.sh synthetic 0.2 100 <here> <here> 50 1 1

# For ex for 50 local steps
./myrun.sh synthetic 0.2 100 50 50 50 1 1

# The logs will be produced in logs/fedavg/artificial/synthetic/<local_step>_<local_step>_lr0.2/r1 directory
```

The generated logs contain `train.csv` and `test.csv` similar to FEMNIST. Additionally, they contain the trained model parameters saved in `trained_model` subfolder. The difference between local and global gradients per round are saved in `grad_diffs.csv` file. Each column in this file corresponds to one client.

## Further customisation details

Each shell script `myrun.sh` mentioned previously in turn runs the `main.py` script in the corresponding directory. 

More functionalities are also supported as detailed below by command line arguments of `main.py` below. Ex. Simulating heterogeneous client local steps by specifying the parameters a and b of the uniform distribution U[a, b].

```
usage: main.py [-h] [-d {femnist}] -traindir TRAINING_DIR -testdir TESTING_DIR -r LEARNING_RATE -b BATCH_SIZE -lb
               LOWER_BOUND -up UPPER_BOUND -l LOGDIR [-n NUM_CLIENTS] [-f FIXED_ROUNDS] [-ee EVALUATE_EVERY]
               [-sm SAVE_MODEL] [-mwf MODEL_WEIGHTS_FILE] -sd SEED

optional arguments:
  -h, --help            show this help message and exit
  -d {femnist}, --dataset {femnist}
                        Dataset on which to run experiment.
  -traindir TRAINING_DIR, --training_dir TRAINING_DIR
                        Absolute path to the directory containing training data.
  -testdir TESTING_DIR, --testing_dir TESTING_DIR
                        Absolute path to the directory containing testing data.
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for training.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training.
  -lb LOWER_BOUND, --lower_bound LOWER_BOUND
                        Lower bound for budget for U(lb, up)
  -up UPPER_BOUND, --upper_bound UPPER_BOUND
                        Uppper bound for budget for U(lb, up)
  -l LOGDIR, --logdir LOGDIR
                        Path to directory for logging. Creates one if not exists.
  -n NUM_CLIENTS, --num_clients NUM_CLIENTS
                        Number of clients to be selected in every round.
  -f FIXED_ROUNDS, --fixed_rounds FIXED_ROUNDS
                        Number of rounds to run if running for fixed rounds.
  -ee EVALUATE_EVERY, --evaluate_every EVALUATE_EVERY
                        Frequency of evaluation on test set.
  -sm SAVE_MODEL, --save_model SAVE_MODEL
                        Set to True to save final global model to log directory. Default is False.
  -mwf MODEL_WEIGHTS_FILE, --model_weights_file MODEL_WEIGHTS_FILE
                        Points to the file containing model weights for same initialisation.
  -sd SEED, --seed SEED
                        Seed for sampling clients and their budgets.
```

Elements to note:
- The number of steps that each client performs locally is sampled from a uniform distribution `[lb, up]`. Thus depending on the values of `-lb` and `-up` parameters, they may perform same or different number of local steps.
- Since model initialisation affects convergence, to conduct reproducible experiments we can specify which initial model to pick via the file path to model weights (`-mwf` parameter). The folder `model_weights` contains a few initial models for every dataset.
- Seed determines the client selection for every communication round.
- Train and test directory paths correspond to how LEAF generates the data. This should generally be `leaf/data/dataset_name/data/{train, test}`.
- The generated logs contain test metrics per `-ee` rounds and train metrics per round. They also contain a tensorboard log (in `/tb` subdirectory) which can be directly used to visualise performance as: `tensorboard --logdir <path to log directory/tb>`.
- Rest all parameters are self-explanatory. 

![Screenshot 2022-05-24 at 16 54 48](https://user-images.githubusercontent.com/24961068/170066838-b7eaaaea-090c-42a5-b106-103f4d611e7f.png)

